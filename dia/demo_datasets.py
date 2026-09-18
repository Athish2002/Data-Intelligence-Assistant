"""
dia/demo_datasets.py
────────────────────
Built-in enterprise demo datasets for zero-friction testing and validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_demo_dataset(dataset_name: str) -> tuple[pd.DataFrame, str, str]:
    """
    Returns a realistic tabular benchmark dataset with suggested ML goal and target.
    Accepts either benchmark IDs ('telecom', 'bank_credit') or full display names.
    """
    key_map = {
        "telecom": "Telecom Customer Churn",
        "bank_credit": "Bank Credit Risk & Default",
        "real_estate": "Real Estate Price Estimation",
        "malformed_retail": "Dirty & Malformed Retail E-Commerce",
        "supply_chain": "Supply Chain Delivery Delay",
        "clinical_sepsis": "Clinical ICU Sepsis Prediction",
    }
    dataset_name = key_map.get(dataset_name.strip().lower(), dataset_name.strip())

    np.random.seed(42)
    n = 600

    if dataset_name == "Telecom Customer Churn":
        tenure = np.random.randint(1, 72, size=n)
        monthly_charges = np.random.uniform(20.0, 120.0, size=n)
        contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
        internet = np.random.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.4, 0.45, 0.15])
        tech_support = np.random.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
        payment = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n)

        # Realistic churn logic
        churn_logits = (
            -1.5
            - 0.04 * tenure
            + 0.03 * (monthly_charges - 60)
            + 0.8 * (contract == "Month-to-month")
            + 0.6 * (tech_support == "No")
            + 0.5 * (internet == "Fiber optic")
        )
        churn_prob = 1.0 / (1.0 + np.exp(-churn_logits))
        churn = (np.random.rand(n) < churn_prob).astype(int)

        df = pd.DataFrame({
            "customer_id": [f"CUST-{1000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": np.round(monthly_charges, 2),
            "contract_type": contract,
            "internet_service": internet,
            "tech_support": tech_support,
            "payment_method": payment,
            "churn_flag": churn,
        })
        goal = "Predict customer churn flag and identify key risk drivers"
        target = "churn_flag"

    elif dataset_name == "Bank Credit Risk & Default":
        income = np.random.exponential(scale=55000, size=n) + 15000
        credit_score = np.random.normal(loc=680, scale=80, size=n).clip(300, 850)
        debt_to_income = np.random.uniform(0.05, 0.65, size=n)
        age = np.random.randint(21, 70, size=n)
        loan_amount = np.random.uniform(2000, 45000, size=n)
        home_ownership = np.random.choice(["RENT", "OWN", "MORTGAGE"], size=n, p=[0.45, 0.15, 0.40])

        default_logits = -2.0 - 0.006 * (credit_score - 600) + 4.0 * (debt_to_income - 0.3) + 0.00003 * loan_amount
        default_prob = 1.0 / (1.0 + np.exp(-default_logits))
        default = (np.random.rand(n) < default_prob).astype(int)

        df = pd.DataFrame({
            "applicant_id": [f"APP-{5000 + i}" for i in range(n)],
            "annual_income": np.round(income, 2),
            "credit_score": np.round(credit_score, 0),
            "debt_to_income_ratio": np.round(debt_to_income, 3),
            "applicant_age": age,
            "loan_amount": np.round(loan_amount, 2),
            "home_ownership": home_ownership,
            "default_risk": default,
        })
        goal = "Predict loan default risk for credit underwriting"
        target = "default_risk"

    elif dataset_name == "Dirty & Malformed Retail E-Commerce":
        # Realistic raw dirty e-commerce dataset
        prices = [f"${np.random.uniform(15, 350):.2f}" for _ in range(n)]
        # Inject occasional currency symbols & units
        prices[0] = "€120.50"
        prices[1] = "£45.00"
        prices[2] = "$1.5k"

        discounts = [f"{np.random.choice([0, 5, 10, 15, 20])}%" for _ in range(n)]
        lifetime_spend = [f"{np.random.uniform(0.5, 15.0):.1f}k" for _ in range(n)]

        statuses = [np.random.choice(["active", "inactive", "N/A", "--", "?", "null"]) for _ in range(n)]
        booleans = [np.random.choice(["yes", "no", "true", "false", "t", "f"]) for _ in range(n)]
        returns = (np.random.rand(n) < 0.22).astype(int)

        df = pd.DataFrame({
            " Customer ID # ": [f"CUST_{i}" for i in range(n)],
            " Item Price ($) ": prices,
            " Discount % ": discounts,
            " Total Customer Lifetime Spend ($) ": lifetime_spend,
            " Account Status (Raw) ": statuses,
            " Prime Member (Yes/No) ": booleans,
            " Return Risk Flag ": returns,
        })
        goal = "Predict return risk flag and sanitize dirty currency and null fields"
        target = "Return Risk Flag"

    elif dataset_name == "Supply Chain Delivery Delay":
        distance = np.random.uniform(100.0, 10000.0, size=n)
        weight = np.random.exponential(scale=25.0, size=n) + 1.0
        customs_days = np.random.poisson(lam=1.5, size=n)
        carrier_rating = np.random.uniform(1.0, 5.0, size=n)
        transport_mode = np.random.choice(["Air", "Sea", "Road", "Rail"], size=n, p=[0.25, 0.35, 0.30, 0.10])
        origin_region = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"], size=n)
        weather_delay_risk = np.random.uniform(0.0, 1.0, size=n)

        delay_logits = (
            -1.8
            + 0.00025 * distance
            + 0.02 * weight
            + 0.6 * customs_days
            - 0.5 * (carrier_rating - 3.0)
            + 1.2 * weather_delay_risk
            + 0.7 * (transport_mode == "Sea")
        )
        delay_prob = 1.0 / (1.0 + np.exp(-delay_logits))
        delay = (np.random.rand(n) < delay_prob).astype(int)

        df = pd.DataFrame({
            "shipment_id": [f"SHIP-{10000 + i}" for i in range(n)],
            "distance_km": np.round(distance, 1),
            "weight_kg": np.round(weight, 2),
            "transport_mode": transport_mode,
            "origin_region": origin_region,
            "customs_days": customs_days,
            "carrier_rating": np.round(carrier_rating, 2),
            "weather_delay_risk": np.round(weather_delay_risk, 3),
            "delay_flag": delay,
        })
        goal = "Predict shipment delivery delay risk across multimodal logistics networks"
        target = "delay_flag"

    elif dataset_name == "Clinical ICU Sepsis Prediction":
        age = np.random.randint(18, 88, size=n)
        icu_hours = np.random.exponential(scale=36, size=n) + 4
        hr = np.random.normal(loc=78, scale=14, size=n).clip(45, 160)
        sbp = np.random.normal(loc=122, scale=18, size=n).clip(60, 200)
        temp = np.random.normal(loc=37.0, scale=0.6, size=n).clip(35.0, 41.0)
        wbc = np.random.normal(loc=7.5, scale=2.8, size=n).clip(2.0, 30.0)
        lactate = np.random.exponential(scale=1.2, size=n) + 0.5
        resp_rate = np.random.normal(loc=16, scale=4, size=n).clip(8, 45)

        # Severe class imbalance (~5-7% positive prevalence)
        sepsis_logits = (
            -4.5
            + 0.025 * (hr - 80)
            - 0.03 * (sbp - 120)
            + 0.8 * (temp - 37.0)
            + 0.12 * (wbc - 8.0)
            + 0.7 * (lactate - 1.5)
            + 0.05 * (resp_rate - 16)
        )
        sepsis_prob = 1.0 / (1.0 + np.exp(-sepsis_logits))
        sepsis = (np.random.rand(n) < sepsis_prob).astype(int)

        # Ensure positive samples exist
        if sepsis.sum() < 10:
            top_risk_idx = np.argsort(sepsis_logits)[-15:]
            sepsis[top_risk_idx] = 1

        df = pd.DataFrame({
            "patient_id": [f"PAT-{8000 + i}" for i in range(n)],
            "age": age,
            "icu_stay_hours": np.round(icu_hours, 1),
            "heart_rate": np.round(hr, 1),
            "systolic_bp": np.round(sbp, 1),
            "body_temp_c": np.round(temp, 2),
            "wbc_count": np.round(wbc, 2),
            "serum_lactate": np.round(lactate, 2),
            "respiratory_rate": np.round(resp_rate, 1),
            "sepsis_target": sepsis,
        })
        goal = "Predict ICU sepsis onset under extreme class imbalance"
        target = "sepsis_target"

    else:  # Real Estate Price Estimation (Regression)
        sqft = np.random.normal(loc=2000, scale=600, size=n).clip(600, 5000)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.2, 0.45, 0.25, 0.05])
        bathrooms = (bedrooms * 0.75 + np.random.uniform(0, 1, size=n)).clip(1, 4)
        age_years = np.random.randint(0, 60, size=n)
        location_tier = np.random.choice(["Urban Core", "Suburban Prime", "Rural"], size=n, p=[0.35, 0.45, 0.20])

        tier_mult = np.where(location_tier == "Urban Core", 1.4, np.where(location_tier == "Suburban Prime", 1.1, 0.8))
        price = (120000 + sqft * 180 + bedrooms * 15000 + bathrooms * 22000 - age_years * 1200) * tier_mult + np.random.normal(0, 25000, size=n)

        df = pd.DataFrame({
            "property_id": [f"PROP-{2000 + i}" for i in range(n)],
            "square_feet": np.round(sqft, 0),
            "num_bedrooms": bedrooms,
            "num_bathrooms": np.round(bathrooms, 1),
            "property_age": age_years,
            "location_tier": location_tier,
            "sale_price": np.round(price, 2),
        })
        goal = "Predict property sale price based on physical and location features"
        target = "sale_price"

    return df, goal, target


DEMO_BENCHMARKS = {
    "telecom": {
        "name": "Telecom Customer Churn",
        "domain": "Telecommunications & SaaS",
        "default_goal": "Predict customer churn flag and identify key risk drivers",
        "default_target": "churn_flag",
        "description": "600 subscriber accounts with contract, payment, charges, and churn flags.",
    },
    "bank_credit": {
        "name": "Bank Credit Risk & Default",
        "domain": "Banking & FinTech Underwriting",
        "default_goal": "Predict applicant credit default risk for loan underwriting",
        "default_target": "default_risk",
        "description": "600 credit applicants with credit scores, income, DTI, and default history.",
    },
    "real_estate": {
        "name": "Real Estate Price Estimation",
        "domain": "Real Estate & Property Valuation",
        "default_goal": "Predict property sale price based on physical and location features",
        "default_target": "sale_price",
        "description": "600 residential homes with square footage, beds, baths, and sale prices.",
    },
    "malformed_retail": {
        "name": "Dirty & Malformed Retail E-Commerce",
        "domain": "Retail, E-Commerce & Data Cleaning",
        "default_goal": "Predict return risk flag and sanitize dirty currency and null fields",
        "default_target": "Return Risk Flag",
        "description": "600 retail records with currencies ($/€/£), multipliers (10k), and missing tokens.",
    },
    "supply_chain": {
        "name": "Supply Chain Delivery Delay",
        "domain": "Supply Chain & Logistics",
        "default_goal": "Predict shipment delivery delay risk across multimodal logistics networks",
        "default_target": "delay_flag",
        "description": "600 multimodal shipments with transport modes, customs times, weather, and delay flags.",
    },
    "clinical_sepsis": {
        "name": "Clinical ICU Sepsis Prediction",
        "domain": "Healthcare & Critical Care",
        "default_goal": "Predict ICU sepsis onset under extreme class imbalance",
        "default_target": "sepsis_target",
        "description": "600 ICU patient records with vitals, biomarkers, and severe class imbalance (~5% sepsis prevalence).",
    },
}

