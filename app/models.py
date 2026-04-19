from sqlalchemy import Column, Integer, String, Text, Date, Time, DateTime
from sqlalchemy.sql import func
from .database import Base


class FIR(Base):
    __tablename__ = "firs"

    id = Column(Integer, primary_key=True, index=True)
    fir_number = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=False)
    station_name = Column(String(120), nullable=False)
    district = Column(String(120), nullable=False, index=True)
    incident_date = Column(Date, nullable=False, index=True)
    incident_time = Column(Time, nullable=True)

    legal_section = Column(String(255), nullable=True)
    crime_type = Column(String(120), nullable=False, index=True)
    priority = Column(String(50), nullable=False, default="Medium")
    status = Column(String(50), nullable=False, default="Open")

    # Analytics-specific fields (inspired by Indian Crimes Dataset)
    weapon_used = Column(String(120), nullable=True)
    victim_age = Column(Integer, nullable=True)
    victim_gender = Column(String(10), nullable=True)
    reported_at = Column(DateTime, nullable=True)

    complainant_name = Column(String(120), nullable=True)
    accused_name = Column(String(120), nullable=True)
    location_text = Column(String(255), nullable=True)

    description = Column(Text, nullable=False)
    raw_fir_text = Column(Text, nullable=True)
    evidence_summary = Column(Text, nullable=True)
    image_path = Column(String(255), nullable=True)
    tags = Column(String(255), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())