
from app.database import SessionLocal, engine, Base
from app.models import FIR
from datetime import datetime, time
import os

def reseed():
    print("Clearing existing data...")
    db = SessionLocal()
    try:
        db.query(FIR).delete()
        db.commit()
    except Exception as e:
        print(f"Error clearing data: {e}")
        db.rollback()

    print("Seeding 250 multi-year analytics records...")
    import random
    from datetime import timedelta

    districts = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]
    crimes = ["Theft", "Assault", "Robbery", "Fraud", "Public Disturbance", "Kidnapping", "Cyber Crime"]
    weapons = ["None", "Knife", "Firearm", "Digital", "Blunt Object", "Physical Force"]
    genders = ["Male", "Female", "Other"]
    
    samples = []
    # Generate data starting from Jan 2024
    start_date = datetime(2024, 1, 1)
    
    # Increase records to 250 for a multi-year spread
    for i in range(1, 251):
        # Random date between Jan 2024 and April 2026 (~830 days)
        random_days = random.randint(0, 830)
        inc_date = start_date + timedelta(days=random_days)
        
        # Random time
        h, m = random.randint(0, 23), random.randint(0, 59)
        inc_time = time(h, m)
        
        # Reported 1-12 hours later
        rep_at = datetime.combine(inc_date, inc_time) + timedelta(hours=random.randint(1, 12))
        
        # Crime characteristics
        ctype = random.choice(crimes)
        priority = "Critical" if ctype in ["Robbery", "Kidnapping"] else random.choice(["High", "Medium", "Low"])
        
        samples.append(FIR(
            fir_number=f"FIR-{inc_date.year}-{i:04d}",
            title=f"Incident Report: {ctype} in {random.choice(districts)}",
            station_name="Police Division-" + str(random.randint(1, 5)),
            district=random.choice(districts),
            incident_date=inc_date.date(),
            incident_time=inc_time,
            reported_at=rep_at,
            legal_section=str(random.randint(300, 500)),
            crime_type=ctype,
            priority=priority,
            status=random.choice(["Open", "Under Investigation", "Closed", "Charge Sheeted"]),
            weapon_used=random.choice(weapons),
            victim_age=random.randint(18, 75),
            victim_gender=random.choice(genders),
            complainant_name="Citizen " + str(i),
            accused_name="Unknown" if random.random() > 0.5 else "Accused " + str(i),
            location_text="Sector " + str(random.randint(1, 20)),
            description=f"Multi-year audit record for {ctype} incident in {inc_date.strftime('%B %Y')}.",
            tags=f"{ctype.lower()}, historical-data"
        ))
    
    db.add_all(samples)
    db.commit()
    db.close()
    print("Reseed of 250 multi-year records complete.")
    
    db.add_all(samples)
    db.commit()
    db.close()
    print("Reseed complete.")

if __name__ == "__main__":
    reseed()
