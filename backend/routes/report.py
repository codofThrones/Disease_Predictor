"""
Diagnosis AI — Report Routes
PDF report generation and download endpoints.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from auth.auth import get_current_user
from models.db_models import User, Prediction, Report
from schemas.schemas import ReportResponse
from reports.report_service import generate_report

logger = logging.getLogger("diagnosis_ai.routes.report")

router = APIRouter(prefix="/api/reports", tags=["Reports"])


def _generate_report_background(
    report_id: int,
    prediction_data: dict,
    filename: str,
    db_url: str,
):
    """Background task to generate a PDF report."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Create a new session for the background task
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(db_url, connect_args=connect_args)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            logger.error(f"Report {report_id} not found for background generation")
            return

        report.status = "generating"
        db.commit()

        # Generate the PDF
        output_path = generate_report(prediction_data, filename)

        report.file_path = output_path
        report.status = "completed"
        db.commit()

        logger.info(f"Report {report_id} generated: {output_path}")

    except Exception as e:
        logger.error(f"Report generation failed for {report_id}: {e}")
        if report:
            report.status = "failed"
            db.commit()
    finally:
        db.close()


@router.post("/{prediction_id}", response_model=ReportResponse, status_code=status.HTTP_202_ACCEPTED)
def create_report(
    prediction_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a PDF report for a prediction (async background task)."""
    # Verify prediction exists and belongs to user
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == current_user.id)
        .first()
    )
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    # Create report record
    report = Report(
        prediction_id=prediction_id,
        status="pending",
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Build prediction data for report
    prediction_data = {
        "predicted_disease": prediction.predicted_disease,
        "confidence": prediction.confidence,
        "model_used": prediction.model_used,
        "top_predictions": prediction.top_predictions_json or [],
        "input_json": prediction.input_json or {},
        "recommendation": "Consult a physician for proper diagnosis.",
        "metadata": (prediction.top_predictions_json[0].get("metadata", {})
                     if prediction.top_predictions_json else {}),
    }

    filename = f"report_{prediction_id}_{report.id}.pdf"

    # Schedule background generation
    from app.config import settings
    background_tasks.add_task(
        _generate_report_background,
        report.id,
        prediction_data,
        filename,
        settings.DATABASE_URL,
    )

    return ReportResponse.model_validate(report)


@router.get("/{report_id}/status", response_model=ReportResponse)
def get_report_status(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Check the status of a report generation."""
    report = (
        db.query(Report)
        .join(Prediction)
        .filter(Report.id == report_id, Prediction.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )
    return ReportResponse.model_validate(report)


@router.get("/{report_id}/download")
def download_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download a generated PDF report."""
    report = (
        db.query(Report)
        .join(Prediction)
        .filter(Report.id == report_id, Prediction.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    if report.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report not ready. Current status: {report.status}",
        )

    if not report.file_path or not Path(report.file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found on disk",
        )

    return FileResponse(
        path=report.file_path,
        media_type="application/pdf",
        filename=Path(report.file_path).name,
    )
