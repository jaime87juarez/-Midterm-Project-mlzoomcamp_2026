# -Midterm-Project-mlzoomcamp_2026
My name is Jaime Juárez and this is a repository of the midterm proyect for the MLZoomcamp cohort 2026.

the proyect will use the Kaggle Repository :https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand



# Comprehensive Guide: Hotel Booking Demand Analysis & Yield Management with Machine Learning

## 1. Dataset Overview
The **Hotel Booking Demand** dataset on Kaggle (originally sourced from data published in *Elsevier* by Antonio, Almeida, and Nunes, 2019) contains hotel reservation records spanning from July 2015 to August 2017.

* **Scale:** Approximately 119,390 rows and 32 columns.
* **Context:** It includes data from two distinct properties: a **City Hotel** and a **Resort Hotel**.
* **Key Features Include:**
  * **Temporal Data:** `arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`, `lead_time` (number of days between booking and arrival).
  * **Stay Details:** `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`.
  * **Commercial Terms:** `adr` (Average Daily Rate / price per night), `deposit_type`, `customer_type`, `market_segment`.
  * **Target/Operational Indicators:** `is_canceled` (whether the booking was cancelled or not), `required_car_parking_spaces`, `total_of_special_requests`.

---

## 2. Defining the Business Problems
In the hospitality industry, hotels face two major challenges that directly impact revenue (Yield Management):
1. **Revenue Uncertainty (Cancellations):** Last-minute cancellations leave rooms empty, destroying potential revenue if unsold inventory cannot be re-allocated.
2. **Dynamic Pricing (Optimal Tariffs):** Setting the right price (`adr`) for a room based on demand patterns, season, and booking lead time to maximize total earnings without losing customers to competitors.

---

## 3. Solving the Problems with Machine Learning

To address these challenges, we can build a dual-pipeline approach combining **Classification** and **Regression**.

### A. Classification Problem: Predicting Booking Cancellations
* **Objective:** Predict whether a customer will cancel their reservation (`is_canceled` = 1) or complete their stay (`is_canceled` = 0) at the time of booking or during the lead-up.
* **Why it matters:** Allows the hotel to implement overbooking strategies, require non-refundable deposits, or re-open inventory proactively.
* **Key Predictive Features:** `lead_time`, `deposit_type`, `previous_cancellations`, `booking_changes`, `customer_type`, `total_of_special_requests`.
* ** Examples of Algorithms:** Logistic Regression (baseline), Random Forest Classifier, or Gradient Boosting (XGBoost/LightGBM).

### B. Regression Problem: Predicting Optimal Pricing (ADR)
* **Objective:** Predict the continuous target variable `adr` (Average Daily Rate) to understand pricing elasticity and suggest dynamic rates.
* **Why it matters:** Helps automated pricing engines adjust room rates dynamically based on demand spikes, lead times, and room types.
* **Key Predictive Features:** `arrival_date_month`, `stays_in_weekend_nights`, `reserved_room_type`, `market_segment`, `adults`.
* **Examples of Algorithms:** Ridge/Lasso Regression, Random Forest Regressor, or XGBoost Regressor.

---

## 4. Model Evaluation Metrics

To ensure our models perform reliably in a real-world production environment, we use specific evaluation metrics tailored to each problem type.

### Evaluation Metrics for Classification (Cancellations)
Since cancellations are often imbalanced, accuracy alone is insufficient.
* **Precision:** Out of all bookings flagged as "likely to cancel", how many actually cancelled? (Minimizes false alarms where a valid guest is penalized).
* **Recall (Sensitivity):** Out of all actual cancellations, how many did our model successfully catch? (Crucial to prevent empty rooms).
* **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced single metric.
* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the model's ability to discriminate between cancelled and non-cancelled reservations across all classification thresholds.

### Evaluation Metrics for Regression (Pricing / ADR)
* **MAE (Mean Absolute Error):** Measures the average magnitude of absolute errors in our predicted room prices (e.g., "our price predictions are off by an average of \$12.50"). Highly interpretable.
* **RMSE (Root Mean Square Error):** Similar to MAE, but squares the errors before averaging, penalizing large pricing mistakes more heavily.
* **R-squared ($R^2$):** Represents the proportion of variance in the `adr` that can be explained by our independent variables (closer to 1.0 is better).

# env
the environment should was created with conda.
then the file requirements.txt was used
I used "pip freeze > requirements.txt" in order to update the requirements.txt
