"""Create a PDF report from logistic_regression_results.json."""

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#D9EAF7")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
LIGHT_GREY = colors.HexColor("#F2F4F6")
MID_GREY = colors.HexColor("#D9DEE3")
TEXT = colors.HexColor("#17202A")
WARNING = colors.HexColor("#FFF2CC")
GREEN = colors.HexColor("#E2F0D9")


def format_number(value):
    return f"{value:,}"


def format_percent(value, digits=4):
    return f"{value:.{digits}f}%"


def format_score(value):
    return f"{value:.6f}"


def table(rows, widths, header=True, right_columns=None):
    right_columns = right_columns or set()
    converted = []
    for row_index, row in enumerate(rows):
        converted_row = []
        for column_index, value in enumerate(row):
            style = styles["TableCell"]
            if row_index == 0 and header:
                style = styles["TableHeader"]
            elif column_index in right_columns:
                style = styles["TableCellRight"]
            converted_row.append(Paragraph(str(value), style))
        converted.append(converted_row)

    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.white),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY),
            ]
        )
    for row_index in range(1 if header else 0, len(converted)):
        if row_index % 2 == (1 if header else 0):
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GREY))
    result.setStyle(TableStyle(commands))
    return result


def section_heading(number, title):
    return Paragraph(f"{number}. {title}", styles["SectionHeading"])


def add_header_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#66717D"))
    canvas.drawString(20 * mm, 10.5 * mm, "Logistic Regression | Initial sample-based evaluation")
    canvas.drawRightString(190 * mm, 10.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report(data, output_path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        title="Logistic Regression Statistical Results Report",
        author="FYP-AdversarialAttacks",
    )
    m = data["metrics"]
    d = data["duplicates"]
    c = data["convergence"]
    t = data["timings_seconds"]
    model = data["model_parameters"]
    original = data["original_class_distribution"]
    sampled = data["sampled_class_distribution"]
    train = data["train_class_distribution"]
    test = data["test_class_distribution"]

    story = [
        Spacer(1, 12 * mm),
        Paragraph("Logistic Regression", styles["ReportTitle"]),
        Paragraph("Statistical Results Report", styles["ReportSubtitle"]),
        Spacer(1, 9 * mm),
        Paragraph(
            "Initial sample-based evaluation of preprocessed network intrusion data",
            styles["ReportLead"],
        ),
        Spacer(1, 10 * mm),
        table(
            [
                ["Experiment item", "Value"],
                ["Model", data["model"]],
                ["Evaluation mode", "Stratified sample without replacement"],
                ["Source files", format_number(data["source_file_count"])],
                ["Source rows", format_number(data["original_rows"])],
                ["Rows evaluated", format_number(data["rows_used"])],
                ["Predictor features", format_number(data["predictor_count"])],
                ["Random state", data["random_state"]],
            ],
            [75 * mm, 95 * mm],
            right_columns={1},
        ),
        Spacer(1, 8 * mm),
        section_heading("1", "Executive summary"),
        Paragraph(
            "This report records one grouped holdout experiment using the "
            "preprocessed CSV files listed in the evaluation result. The model "
            "was trained with balanced class weights and evaluated on a "
            "duplicate-safe 80/20 holdout.",
            styles["Body"]),
        Spacer(1, 4 * mm),
        table(
            [
                ["Headline metric", "Result"],
                ["Accuracy", format_score(m["accuracy"])],
                ["Balanced accuracy", format_score(m["balanced_accuracy"])],
                ["Macro F1", format_score(m["macro_f1"])],
                ["Normal specificity", format_score(m["normal_recall_specificity"])],
                ["Attack recall / sensitivity", format_score(m["attack_recall_sensitivity"])],
            ],
            [75 * mm, 95 * mm],
            right_columns={1},
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "Important: the Logistic Regression solver did not converge within "
            f"{model['max_iter']:,} iterations. The reported values are therefore "
            "the configured baseline result, not a converged optimum.",
            styles["Warning"]),
        PageBreak(),
        section_heading("2", "Data and class distribution"),
        Paragraph(
            "The evaluator validated all selected model CSVs before sampling. "
            "The sample preserved the source class proportions as closely as "
            "possible and used random state 42.",
            styles["Body"]),
        Spacer(1, 4 * mm),
        table(
            [
                ["Data statistic", "Normal (0)", "Attack (1)", "Total"],
                [
                    "Original source",
                    f"{format_number(original['0']['count'])} ({original['0']['percentage']:.4f}%)",
                    f"{format_number(original['1']['count'])} ({original['1']['percentage']:.4f}%)",
                    format_number(data["original_rows"]),
                ],
                [
                    "Evaluated sample",
                    f"{format_number(sampled['0']['count'])} ({sampled['0']['percentage']:.4f}%)",
                    f"{format_number(sampled['1']['count'])} ({sampled['1']['percentage']:.4f}%)",
                    format_number(data["rows_used"]),
                ],
                [
                    "Training split",
                    f"{format_number(train['0']['count'])} ({train['0']['percentage']:.4f}%)",
                    f"{format_number(train['1']['count'])} ({train['1']['percentage']:.4f}%)",
                    format_number(data["train_rows"]),
                ],
                [
                    "Test split",
                    f"{format_number(test['0']['count'])} ({test['0']['percentage']:.4f}%)",
                    f"{format_number(test['1']['count'])} ({test['1']['percentage']:.4f}%)",
                    format_number(data["test_rows"]),
                ],
            ],
            [42 * mm, 43 * mm, 43 * mm, 42 * mm],
            right_columns={1, 2, 3},
        ),
        Spacer(1, 8 * mm),
        section_heading("3", "Holdout and duplicate statistics"),
        Paragraph(
            f"Only the first split from {data['split_strategy']} was used. "
            "This is one holdout, not cross-validation. Exact numeric predictor "
            "equality determined groups; label and type were excluded.",
            styles["Body"]),
        Spacer(1, 4 * mm),
        table(
            [
                ["Split statistic", "Value"],
                ["Training proportion", format_percent(data["train_percentage"])],
                ["Test proportion", format_percent(data["test_percentage"])],
                [
                    "Deviation from target (train / test)",
                    f"{data['train_deviation_percentage_points']:+.4f} / "
                    f"{data['test_deviation_percentage_points']:+.4f} percentage points",
                ],
            ],
            [90 * mm, 80 * mm],
            right_columns={1},
        ),
        Spacer(1, 4 * mm),
        table(
            [
                ["Duplicate statistic within the sample", "Value"],
                ["Distinct predictor groups", format_number(d["distinct_predictor_groups"])],
                ["Groups containing duplicates", format_number(d["duplicate_groups"])],
                [
                    "Repeated rows beyond first",
                    f"{format_number(d['repeated_rows_beyond_first'])} "
                    f"({100 * d['repeated_rows_beyond_first'] / data['rows_used']:.4f}%)",
                ],
                ["Largest group", f"{format_number(d['largest_group_rows'])} rows"],
                ["Conflicting-label groups", format_number(d["conflicting_label_groups"])],
                ["Rows in conflicting-label groups", format_number(d["rows_in_conflicting_label_groups"])],
                ["Groups crossing train/test", format_number(d["crossing_groups"])],
            ],
            [90 * mm, 80 * mm],
            right_columns={1},
        ),
        PageBreak(),
        section_heading("4", "Model and convergence"),
        Paragraph(
            "The baseline used scikit-learn LogisticRegression with balanced "
            "class weights. No additional scaling, imputation, encoding, feature "
            "selection, resampling, or parameter tuning was performed by the evaluator.",
            styles["Body"]),
        Spacer(1, 4 * mm),
        table(
            [
                ["Model parameter", "Value"],
                ["Estimator", data["model"]],
                ["Solver", model["solver"]],
                ["Class weight", model["class_weight"]],
                ["Maximum iterations", format_number(model["max_iter"])],
                ["C", model["C"]],
                ["Tolerance", model["tol"]],
                ["Random state", model["random_state"]],
                ["scikit-learn version", data["sklearn_version"]],
                ["NumPy version", data["numpy_version"]],
                ["pandas version", data["pandas_version"]],
            ],
            [90 * mm, 80 * mm],
            right_columns={1},
        ),
        Spacer(1, 7 * mm),
        Paragraph("Convergence status", styles["Subheading"]),
        table(
            [
                ["Convergence item", "Value"],
                ["Converged", "No"],
                ["Iterations used", format_number(c["iterations"][0])],
                ["Warning", "LBFGS reached the configured iteration limit"],
            ],
            [90 * mm, 80 * mm],
            right_columns={1},
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "The warning recommends increasing max_iter or scaling the data. "
            "Those changes were not applied because this report documents the "
            "configured baseline exactly.",
            styles["Warning"]),
        Spacer(1, 8 * mm),
        section_heading("5", "Confusion matrix"),
        Paragraph(
            "Rows are true labels and columns are predicted labels, in the order [0, 1].",
            styles["Body"]),
        Spacer(1, 3 * mm),
        table(
            [
                ["", "Predicted normal (0)", "Predicted attack (1)"],
                ["True normal (0)", format_number(m["confusion_matrix"][0][0]), format_number(m["confusion_matrix"][0][1])],
                ["True attack (1)", format_number(m["confusion_matrix"][1][0]), format_number(m["confusion_matrix"][1][1])],
            ],
            [55 * mm, 57.5 * mm, 57.5 * mm],
            right_columns={1, 2},
        ),
        PageBreak(),
        section_heading("6", "Classification metrics"),
        table(
            [
                ["Metric", "Value", "Interpretation"],
                ["Accuracy", format_score(m["accuracy"]), "Overall correct predictions"],
                ["Balanced accuracy", format_score(m["balanced_accuracy"]), "Mean recall across both classes"],
                ["Attack precision", format_score(m["attack_precision"]), "Predicted attacks that were attacks"],
                ["Attack recall / sensitivity", format_score(m["attack_recall"]), "Attacks correctly detected"],
                ["Attack F1", format_score(m["attack_f1"]), "Attack precision-recall harmonic mean"],
                ["Macro precision", format_score(m["macro_precision"]), "Unweighted precision across classes"],
                ["Macro recall", format_score(m["macro_recall"]), "Unweighted recall across classes"],
                ["Macro F1", format_score(m["macro_f1"]), "Unweighted F1 across classes"],
                ["Normal specificity", format_score(m["normal_recall_specificity"]), "Normal records correctly rejected as attacks"],
                ["Attack false-negative rate", format_score(m["attack_false_negative_rate"]), "Attacks missed by the model"],
                ["Normal false-positive rate", format_score(m["normal_false_positive_rate"]), "Normal records flagged as attacks"],
            ],
            [64 * mm, 34 * mm, 72 * mm],
            right_columns={1},
        ),
        Spacer(1, 8 * mm),
        section_heading("7", "Evaluation timings"),
        table(
            [
                ["Stage", "Seconds"],
                ["Discovery, validation, loading", f"{t['discovery_validation_loading']:.3f}"],
                ["Sampling", f"{t['sampling']:.3f}"],
                ["Duplicate grouping and splitting", f"{t['duplicate_grouping_splitting']:.3f}"],
                ["Training", f"{t['training']:.3f}"],
                ["Prediction", f"{t['prediction']:.3f}"],
                ["Total evaluation", f"{t['total']:.3f}"],
            ],
            [120 * mm, 50 * mm],
            right_columns={1},
        ),
        Spacer(1, 8 * mm),
        section_heading("8", "Limitations and interpretation"),
        *[
            Paragraph(f"- {item}", styles["ReportBullet"])
            for item in data["limitations"]
        ],
        Spacer(1, 6 * mm),
        Paragraph(
            "This report is generated directly from the recorded JSON result and "
            "does not claim adversarial robustness. Macro F1, balanced accuracy, "
            "and both class recalls should be considered alongside the severe "
            "class imbalance and the non-converged solver status.",
            styles["Closing"]),
    ]
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results" / "logistic_regression_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "logistic_regression_results_report.pdf",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_report(data, args.output)
    print(f"Report written to {args.output}")


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=25, leading=30, alignment=TA_CENTER, textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="ReportSubtitle", parent=styles["Normal"], fontName="Helvetica-Bold",
    fontSize=17, leading=22, alignment=TA_CENTER, textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="ReportLead", parent=styles["Normal"], fontSize=11, leading=16,
    alignment=TA_CENTER, textColor=TEXT,
))
styles.add(ParagraphStyle(
    name="SectionHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=15, leading=19, spaceBefore=2, spaceAfter=8, textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="Subheading", parent=styles["Heading3"], fontName="Helvetica-Bold",
    fontSize=11, leading=14, textColor=NAVY, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Body", parent=styles["BodyText"], fontSize=9.5, leading=13,
    textColor=TEXT, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="ReportBullet", parent=styles["BodyText"], fontSize=9.2, leading=13,
    leftIndent=8, firstLineIndent=-8, textColor=TEXT, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="Warning", parent=styles["BodyText"], fontSize=9.3, leading=13,
    backColor=WARNING, borderColor=colors.HexColor("#D6B656"),
    borderWidth=0.5, borderPadding=7, textColor=TEXT, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Closing", parent=styles["BodyText"], fontSize=9.5, leading=13,
    backColor=LIGHT_BLUE, borderColor=BLUE, borderWidth=0.5,
    borderPadding=7, textColor=TEXT,
))
styles.add(ParagraphStyle(
    name="TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold",
    fontSize=8.4, leading=10.5, textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="TableCell", parent=styles["Normal"], fontSize=8.4,
    leading=10.5, textColor=TEXT,
))
styles.add(ParagraphStyle(
    name="TableCellRight", parent=styles["Normal"], fontSize=8.4,
    leading=10.5, alignment=TA_RIGHT, textColor=TEXT,
))


if __name__ == "__main__":
    main()
