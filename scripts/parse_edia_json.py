#!/usr/bin/env python3
"""
Parse density-fitness (EDIA) JSON output to CSV format.
Usage: python parse_edia_json.py <input_file> [output_file]
"""

import json
import sys

import pandas as pd
from loguru import logger

import pandas as pd
from loguru import logger


def parse_protein_data(file_path):
    """
    Parse the JSON file containing protein residue data.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        pandas.DataFrame: Parsed data as a DataFrame
    """
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()

            # Handle malformed JSON (missing opening/closing brackets)
            if not content.startswith("["):
                content = "[" + content
            if not content.endswith("]"):
                content = content + "]"

            data = json.loads(content)

        df = pd.DataFrame(data)

        # Flatten the nested 'pdb' column if it exists
        if "pdb" in df.columns:
            pdb_df = pd.json_normalize(df["pdb"])
            pdb_df.columns = ["pdb_" + col for col in pdb_df.columns]
            df = pd.concat([df.drop("pdb", axis=1), pdb_df], axis=1)

        return df

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        return None


def save_to_csv(df, output_path):
    """
    Save the DataFrame to CSV.

    Args:
        df (pandas.DataFrame): Data to save
        output_path (str): Output CSV file path
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python parse_edia_json.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    df = parse_protein_data(input_file)

    if df is not None:
        if output_file:
            save_to_csv(df, output_file)
        else:
            # Print summary if no output file specified
            logger.info(f"Total residues: {len(df)}")
            logger.info(f"Columns: {list(df.columns)}")
    else:
        logger.error("Failed to parse the file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
