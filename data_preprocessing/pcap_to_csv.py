"""Export Ethernet PCAP/PCAPNG captures as bidirectional flow feature CSVs.

This is a lightweight flow exporter, not a full CICFlowMeter implementation.
It supports Ethernet and Linux-cooked IPv4/IPv6 traffic, with other IP
protocols represented with zero ports. Linux ETH_P_ALL raw IPv4-shaped packets
are retained, including malformed-version headers that may be attack examples.
Non-IP packets are skipped.
"""

from __future__ import annotations

import argparse
import heapq
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import dpkt


CSV_COLUMNS = [
    "src_port", "dst_port", "protocol_number", "ip_version",
    "malformed_ip_packets", "duration_seconds", "src_packets",
    "dst_packets", "src_ip_bytes", "dst_ip_bytes", "total_packets",
    "total_ip_bytes", "mean_packet_bytes", "std_packet_bytes",
    "min_packet_bytes", "max_packet_bytes", "mean_interarrival_seconds",
    "max_interarrival_seconds", "tcp_syn_count", "tcp_ack_count",
    "tcp_fin_count", "tcp_rst_count", "tcp_psh_count", "tcp_urg_count",
    "label", "type",
]


@dataclass
class Flow:
    """Running aggregate for one bidirectional endpoint/protocol tuple."""

    flow_id: int
    key: tuple
    first_src: tuple[str, int]
    first_dst: tuple[str, int]
    protocol_number: int
    ip_version: int
    start_time: float
    last_seen: float
    last_packet_time: float
    src_packets: int = 0
    dst_packets: int = 0
    src_ip_bytes: int = 0
    dst_ip_bytes: int = 0
    malformed_ip_packets: int = 0
    packet_count: int = 0
    packet_size_mean: float = 0.0
    packet_size_m2: float = 0.0
    min_packet_bytes: int | None = None
    max_packet_bytes: int = 0
    interarrival_count: int = 0
    interarrival_sum: float = 0.0
    max_interarrival: float = 0.0
    tcp_syn_count: int = 0
    tcp_ack_count: int = 0
    tcp_fin_count: int = 0
    tcp_rst_count: int = 0
    tcp_psh_count: int = 0
    tcp_urg_count: int = 0

    def add_packet(
        self,
        timestamp: float,
        src: tuple[str, int],
        packet_bytes: int,
        tcp_flags: int | None,
        malformed_ip: bool,
    ) -> None:
        forward = src == self.first_src
        if forward:
            self.src_packets += 1
            self.src_ip_bytes += packet_bytes
        else:
            self.dst_packets += 1
            self.dst_ip_bytes += packet_bytes
        self.malformed_ip_packets += malformed_ip

        self.packet_count += 1
        delta = packet_bytes - self.packet_size_mean
        self.packet_size_mean += delta / self.packet_count
        self.packet_size_m2 += delta * (packet_bytes - self.packet_size_mean)
        self.min_packet_bytes = packet_bytes if self.min_packet_bytes is None else min(self.min_packet_bytes, packet_bytes)
        self.max_packet_bytes = max(self.max_packet_bytes, packet_bytes)

        if timestamp >= self.last_packet_time:
            interarrival = timestamp - self.last_packet_time
            if self.packet_count > 1:
                self.interarrival_count += 1
                self.interarrival_sum += interarrival
                self.max_interarrival = max(self.max_interarrival, interarrival)
        self.last_packet_time = max(self.last_packet_time, timestamp)
        self.start_time = min(self.start_time, timestamp)
        self.last_seen = max(self.last_seen, timestamp)

        if tcp_flags is not None:
            self.tcp_syn_count += bool(tcp_flags & dpkt.tcp.TH_SYN)
            self.tcp_ack_count += bool(tcp_flags & dpkt.tcp.TH_ACK)
            self.tcp_fin_count += bool(tcp_flags & dpkt.tcp.TH_FIN)
            self.tcp_rst_count += bool(tcp_flags & dpkt.tcp.TH_RST)
            self.tcp_psh_count += bool(tcp_flags & dpkt.tcp.TH_PUSH)
            self.tcp_urg_count += bool(tcp_flags & dpkt.tcp.TH_URG)

    def to_row(self, label: str, attack_type: str) -> dict:
        duration = max(0.0, self.last_seen - self.start_time)
        return {
            "src_port": self.first_src[1],
            "dst_port": self.first_dst[1],
            "protocol_number": self.protocol_number,
            "ip_version": self.ip_version,
            "malformed_ip_packets": self.malformed_ip_packets,
            "duration_seconds": round(duration, 6),
            "src_packets": self.src_packets,
            "dst_packets": self.dst_packets,
            "src_ip_bytes": self.src_ip_bytes,
            "dst_ip_bytes": self.dst_ip_bytes,
            "total_packets": self.packet_count,
            "total_ip_bytes": self.src_ip_bytes + self.dst_ip_bytes,
            "mean_packet_bytes": round(self.packet_size_mean, 6),
            "std_packet_bytes": round((self.packet_size_m2 / self.packet_count) ** 0.5, 6),
            "min_packet_bytes": self.min_packet_bytes or 0,
            "max_packet_bytes": self.max_packet_bytes,
            "mean_interarrival_seconds": round(
                self.interarrival_sum / self.interarrival_count, 6
            ) if self.interarrival_count else 0.0,
            "max_interarrival_seconds": round(self.max_interarrival, 6),
            "tcp_syn_count": self.tcp_syn_count,
            "tcp_ack_count": self.tcp_ack_count,
            "tcp_fin_count": self.tcp_fin_count,
            "tcp_rst_count": self.tcp_rst_count,
            "tcp_psh_count": self.tcp_psh_count,
            "tcp_urg_count": self.tcp_urg_count,
            "label": label,
            "type": attack_type,
        }


def _ip_text(raw: bytes, version: int) -> str:
    family = socket.AF_INET if version == 4 else socket.AF_INET6
    return socket.inet_ntop(family, raw)


def _packet_parts(frame: bytes, linktype: int):
    network = frame
    raw_all_protocols = False
    if linktype == dpkt.pcap.DLT_EN10MB:
        network = dpkt.ethernet.Ethernet(frame).data
        while isinstance(network, dpkt.ethernet.VLANtag8021Q):
            network = network.data
    elif linktype == dpkt.pcap.DLT_LINUX_SLL:
        cooked = dpkt.sll.SLL(frame)
        network = cooked.data
        raw_all_protocols = cooked.ethtype == 3
    elif linktype == getattr(dpkt.pcap, "DLT_LINUX_SLL2", 276):
        sll2_class = getattr(dpkt.sll, "SLL2", None)
        if sll2_class is None:
            raise ValueError("This dpkt version does not support Linux cooked v2 packets")
        network = sll2_class(frame).data
    else:
        raise ValueError(f"Unsupported capture link type: {linktype}")

    malformed_ip = False
    if raw_all_protocols and isinstance(network, bytes) and len(network) >= 20:
        # Linux ETH_P_ALL records can contain raw IPv4 bytes even if the version
        # nibble was altered. Keep these flows and expose the malformed header.
        version_nibble = network[0] >> 4
        ihl = (network[0] & 0x0F) * 4
        protocol_number = network[9]
        if protocol_number not in (
            dpkt.ip.IP_PROTO_TCP, dpkt.ip.IP_PROTO_UDP, dpkt.ip.IP_PROTO_ICMP
        ):
            return None
        src_ip = _ip_text(network[12:16], 4)
        dst_ip = _ip_text(network[16:20], 4)
        malformed_ip = version_nibble != 4 or ihl < 20 or ihl > len(network)
        version = 4 if not malformed_ip else 0
        packet_bytes = int.from_bytes(network[2:4], "big")
        if packet_bytes <= 0 or packet_bytes > len(network):
            packet_bytes = len(network)
        transport_offset = ihl if not malformed_ip else 20
        transport = network[transport_offset:]
        src_port = dst_port = 0
        tcp_flags = None
        try:
            if protocol_number == dpkt.ip.IP_PROTO_TCP and len(transport) >= 20:
                tcp = dpkt.tcp.TCP(transport)
                src_port, dst_port, tcp_flags = tcp.sport, tcp.dport, tcp.flags
            elif protocol_number == dpkt.ip.IP_PROTO_UDP and len(transport) >= 8:
                udp = dpkt.udp.UDP(transport)
                src_port, dst_port = udp.sport, udp.dport
        except (dpkt.UnpackError, ValueError):
            pass

        src, dst = (src_ip, int(src_port)), (dst_ip, int(dst_port))
        key = (version, protocol_number, tuple(sorted((src, dst))))
        return key, src, dst, version, protocol_number, int(packet_bytes), tcp_flags, malformed_ip
    elif isinstance(network, dpkt.ip.IP):
        version = 4
        src_ip = _ip_text(network.src, version)
        dst_ip = _ip_text(network.dst, version)
        protocol_number = network.p
        packet_bytes = network.len or len(network)
        transport = network.data
    elif isinstance(network, dpkt.ip6.IP6):
        version = 6
        src_ip = _ip_text(network.src, version)
        dst_ip = _ip_text(network.dst, version)
        protocol_number = network.nxt
        packet_bytes = 40 + network.plen
        transport = network.data
    else:
        return None

    src_port = dst_port = 0
    tcp_flags = None
    if protocol_number == dpkt.ip.IP_PROTO_TCP and isinstance(transport, dpkt.tcp.TCP):
        src_port, dst_port = transport.sport, transport.dport
        tcp_flags = transport.flags
    elif protocol_number == dpkt.ip.IP_PROTO_UDP and isinstance(transport, dpkt.udp.UDP):
        src_port, dst_port = transport.sport, transport.dport

    src = (src_ip, int(src_port))
    dst = (dst_ip, int(dst_port))
    endpoints = tuple(sorted((src, dst)))
    key = (version, protocol_number, endpoints)
    return key, src, dst, version, protocol_number, int(packet_bytes), tcp_flags, malformed_ip


def _make_reader(source):
    magic = source.read(4)
    source.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        reader = dpkt.pcapng.Reader(source)
    else:
        reader = dpkt.pcap.Reader(source)
    linktype = reader.datalink()
    supported = {
        dpkt.pcap.DLT_EN10MB,
        dpkt.pcap.DLT_LINUX_SLL,
        getattr(dpkt.pcap, "DLT_LINUX_SLL2", 276),
    }
    if linktype not in supported:
        raise ValueError(f"Unsupported capture link type: {linktype}")
    return reader, linktype


def export_pcap(
    pcap_path: Path,
    output_path: Path,
    idle_timeout: float = 60.0,
    label: int | None = None,
    attack_type: str | None = None,
) -> dict:
    """Extract bidirectional TCP/UDP/IP flows and write a clean feature CSV."""
    import csv

    pcap_path, output_path = Path(pcap_path), Path(output_path)
    if not pcap_path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
    if pcap_path.suffix.casefold() not in {".pcap", ".pcapng"}:
        raise ValueError("Input must have a .pcap or .pcapng extension")
    if idle_timeout <= 0:
        raise ValueError("Idle timeout must be greater than zero")
    if label not in (None, 0, 1):
        raise ValueError("Label must be 0 (normal), 1 (attack), or omitted")
    if label is None and attack_type:
        raise ValueError("Supply --label when supplying --type")
    if label == 0 and attack_type not in (None, "normal"):
        raise ValueError("Normal traffic must use --type normal")

    label_text = "" if label is None else str(label)
    type_text = "" if label is None else (attack_type or ("normal" if label == 0 else "attack"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active: dict[int, Flow] = {}
    flow_by_key: dict[tuple, int] = {}
    expiry_heap: list[tuple[float, int, float]] = []
    next_flow_id = 0
    max_seen_time = float("-inf")
    packets_seen = 0
    packets_skipped = 0
    flows_written = 0
    malformed_ip_packets = 0
    started = time.perf_counter()

    with pcap_path.open("rb") as capture, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        def write_flow(flow: Flow) -> None:
            nonlocal flows_written, malformed_ip_packets
            writer.writerow(flow.to_row(label_text, type_text))
            flows_written += 1
            malformed_ip_packets += flow.malformed_ip_packets

        reader, linktype = _make_reader(capture)
        for timestamp, frame in reader:
            packets_seen += 1
            try:
                parts = _packet_parts(frame, linktype)
            except (dpkt.UnpackError, ValueError, IndexError):
                packets_skipped += 1
                continue
            if parts is None:
                packets_skipped += 1
                continue

            key, src, dst, ip_version, protocol_number, packet_bytes, tcp_flags, malformed_ip = parts
            max_seen_time = max(max_seen_time, float(timestamp))
            cutoff = max_seen_time - idle_timeout
            while expiry_heap and expiry_heap[0][0] <= cutoff:
                _, flow_id, scheduled_last_seen = heapq.heappop(expiry_heap)
                flow = active.get(flow_id)
                if flow is None:
                    continue
                if flow.last_seen > scheduled_last_seen:
                    heapq.heappush(
                        expiry_heap,
                        (flow.last_seen + idle_timeout, flow_id, flow.last_seen),
                    )
                    continue
                active.pop(flow_id)
                flow_by_key.pop(flow.key, None)
                write_flow(flow)

            flow_id = flow_by_key.get(key)
            flow = active.get(flow_id) if flow_id is not None else None
            if flow is None:
                next_flow_id += 1
                flow_id = next_flow_id
                flow = Flow(
                    flow_id=flow_id,
                    key=key,
                    first_src=src,
                    first_dst=dst,
                    protocol_number=protocol_number,
                    ip_version=ip_version,
                    start_time=float(timestamp),
                    last_seen=float(timestamp),
                    last_packet_time=float(timestamp),
                )
                active[flow_id] = flow
                flow_by_key[key] = flow_id
                heapq.heappush(
                    expiry_heap,
                    (flow.last_seen + idle_timeout, flow_id, flow.last_seen),
                )
            flow.add_packet(float(timestamp), src, packet_bytes, tcp_flags, malformed_ip)

        for flow in sorted(active.values(), key=lambda item: item.start_time):
            write_flow(flow)

    return {
        "input_file": pcap_path.name,
        "output_file": str(output_path),
        "packets_seen": packets_seen,
        "packets_skipped": packets_skipped,
        "flows_written": flows_written,
        "malformed_ip_packets": malformed_ip_packets,
        "label": label_text or None,
        "type": type_text or None,
        "idle_timeout_seconds": idle_timeout,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap", type=Path, help="Input Ethernet .pcap or .pcapng")
    parser.add_argument("output", type=Path, help="Output flow-feature CSV")
    parser.add_argument("--idle-timeout", type=float, default=60.0,
                        help="Start a new flow after this many idle seconds (default: 60)")
    parser.add_argument("--label", type=int, choices=(0, 1),
                        help="Optional ground-truth label: 0 normal, 1 attack")
    parser.add_argument("--type", dest="attack_type",
                        help="Optional ground-truth type (for example scanning or ddos)")
    args = parser.parse_args()
    result = export_pcap(
        args.pcap, args.output, args.idle_timeout, args.label, args.attack_type
    )
    print(
        f"Exported {result['flows_written']:,} flows from {result['packets_seen']:,} packets "
        f"({result['packets_skipped']:,} skipped, "
        f"{result['malformed_ip_packets']:,} malformed-IP packets retained) to {result['output_file']}",
        flush=True,
    )
    if result["label"] is None:
        print("Rows are unlabeled; supply --label/--type only when ground truth is known.", flush=True)


if __name__ == "__main__":
    main()
