from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import typer
from AT1_AML.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from nba_prep.cleaning import clean_height, clean_year, calculate_age
from nba_prep.features import (
    calculate_ast_tov_ratio,
    calculate_true_shooting_percentage,
    calculate_defensive_impact,
    calculate_scoring_efficiency,
)

app = typer.Typer()

def process_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans and engineers features for a raw dataframe.
    """
    processed = df.copy()
    
    logger.info("Cleaning height and year columns...")
    processed["ht_inches"] = processed["ht"].apply(clean_height)
    processed["yr_clean"] = processed["yr"].apply(clean_year)
    
    logger.info("Calculating player age during the season...")
    # Clean DOB to avoid float formats if any, then calculate age
    processed["age"] = processed.apply(
        lambda r: calculate_age(r["year"], r["dob"]), axis=1
    )
    
    logger.info("Calculating advanced basketball features...")
    
    # Estimate turnovers from ast and ast/tov
    tov_est = np.where(
        (processed["ast/tov"] > 0) & (processed["ast/tov"].notna()),
        processed["ast"] / processed["ast/tov"],
        0.0
    )
    processed["ast_tov_ratio_clean"] = calculate_ast_tov_ratio(processed["ast"], tov_est)
    
    # Calculate True Shooting Percentage
    fga = processed["twoPA"] + processed["TPA"]
    processed["true_shooting_per_clean"] = calculate_true_shooting_percentage(
        processed["pts"], fga, processed["FTA"]
    )
    
    # Calculate Defensive Impact Score
    processed["defensive_impact_score"] = calculate_defensive_impact(
        processed["stl"], processed["blk"], processed["dreb"]
    )
    
    # Calculate Scoring Efficiency
    processed["scoring_efficiency_score"] = calculate_scoring_efficiency(
        processed["pts"], processed["Min_per"]
    )
    
    # Calculate Usage-Offensive Rating Interaction
    processed["usage_ortg_interaction"] = processed["usg"] * processed["ORtg"]
    
    return processed

@app.command()
def main(
    train_input: Path = RAW_DATA_DIR / "train.csv",
    test_input: Path = RAW_DATA_DIR / "test.csv",
    train_output: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    test_output: Path = PROCESSED_DATA_DIR / "test_processed.csv",
):
    logger.info("Loading raw datasets...")
    train = pd.read_csv(train_input, low_memory=False)
    test = pd.read_csv(test_input, low_memory=False)
    
    logger.info("Processing train set...")
    train_processed = process_dataframe(train, is_train=True)
    
    logger.info("Processing test set...")
    test_processed = process_dataframe(test, is_train=False)
    
    logger.info("Saving processed datasets...")
    train_processed.to_csv(train_output, index=False)
    test_processed.to_csv(test_output, index=False)
    
    logger.success("Preprocessing and feature engineering complete.")

if __name__ == "__main__":
    app()
