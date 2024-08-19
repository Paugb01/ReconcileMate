import pandas as pd
import numpy as np
from itertools import combinations
import logging
import os
import csv
import streamlit as st
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_file(file, header_row=0):
    if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file, header=header_row)
    elif file.name.endswith('.csv'):
        delimiter = detect_csv_delimiter(file)
        return pd.read_csv(file, header=header_row, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file format: {file.name}")

def detect_csv_delimiter(file, num_chars=1024):
    sniffer = csv.Sniffer()
    sample = file.read(num_chars).decode('utf-8', errors='ignore')
    file.seek(0)
    try:
        delimiter = sniffer.sniff(sample).delimiter
        if delimiter not in [',', ';', '\t', '|']:
            raise ValueError("Unrecognized delimiter. Please specify the delimiter.")
        return delimiter
    except csv.Error:
        raise ValueError("Could not automatically determine the delimiter. Please specify the delimiter.")

def prompt_columns(df, file_type):
    st.write(f"\nFile Type: {file_type}")
    st.write("Available Columns:", list(df.columns))
    
    date_col = st.selectbox("Which column represents the Date?", df.columns)
    
    if file_type in ['Accounting', 'Bank']:
        if file_type == 'Accounting':
            debit_col = st.selectbox("Which column represents Debit?", df.columns)
            credit_col = st.selectbox("Which column represents Credit?", df.columns)
            return date_col, debit_col, credit_col
        elif file_type == 'Bank':
            amount_col = st.selectbox("Which column represents Amount?", df.columns)
            return date_col, amount_col
    else:
        raise ValueError("Unsupported file type")

def convert_to_float(value):
    if pd.isna(value) or (isinstance(value, str) and value.strip() == ''):
        return np.nan
    if isinstance(value, str):
        try:
            return float(value.replace(',', '.'))
        except ValueError as e:
            logging.error(f"Error converting value {value} to float: {e}")
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        logging.error(f"Unexpected data type for conversion: {type(value)}")
        return np.nan

def process_bank(df, date_col, amount_col):
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors='coerce')
    df[amount_col] = df[amount_col].apply(convert_to_float)
    df.rename(columns={amount_col: 'Amount_Bank', date_col: 'Transaction_Date'}, inplace=True)
    df['concat'] = df['Transaction_Date'].astype(str) + ' ' + df['Amount_Bank'].astype(str)
    logging.info(f"Bank: {len(df)} records processed.")
    return df

def process_ledger(df, date_col, debit_col, credit_col):
    df[debit_col] = df[debit_col].apply(convert_to_float)
    df[credit_col] = df[credit_col].apply(convert_to_float)
    df[debit_col] = df[debit_col].fillna(0)
    df[credit_col] = df[credit_col].fillna(0)
    df['Amount_Accounting'] = df[debit_col] - df[credit_col]
    df.dropna(subset=[date_col], inplace=True)
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors='coerce')
    df.rename(columns={date_col: 'Transaction_Date'}, inplace=True)
    df['concat'] = df['Transaction_Date'].astype(str) + ' ' + df['Amount_Accounting'].astype(str)
    logging.info(f"Accounting: {len(df)} records processed.")
    return df

def crossing(df1, df2, col1='concat', col2='concat'):
    try:
        conc = pd.merge(df1, df2, left_on=col1, right_on=col2)
        not_conc_1 = df1[~df1[col1].isin(conc[col1])]
        not_conc_2 = df2[~df2[col2].isin(conc[col2])]
        conc_1 = df1[df1[col1].isin(conc[col1])].index.values.tolist()
        conc_2 = df2[df2[col2].isin(conc[col2])].index.values.tolist()
        logging.info(f"Initial Reconciliation: {len(conc)} transactions reconciled.")
        return not_conc_1, not_conc_2, conc_1, conc_2
    except Exception as e:
        logging.error(f"Error during initial reconciliation: {e}")
        raise

def find_matches_efficiently(og_df1, og_df2, not_conc_1, not_conc_2, conc_1, conc_2, col1='Amount_Accounting', col2='Amount_Bank', max_comb_size=3):
    matched_df1_indices = conc_1
    matched_df2_indices = conc_2

    not_conc_2_sorted = not_conc_2.sort_values(by=col2, key=abs)

    for idx_1, row_1 in not_conc_1.iterrows():
        amount_1 = row_1[col1]
        exact_match = not_conc_2_sorted[not_conc_2_sorted[col2] == amount_1]

        if not exact_match.empty:
            idx_2 = exact_match.index[0]
            matched_df1_indices.append(idx_1)
            matched_df2_indices.append(idx_2)
            not_conc_2_sorted = not_conc_2_sorted.drop(idx_2)
            continue

    for r in range(2, max_comb_size + 1):
        for idx_1, row_1 in not_conc_1.iterrows():
            amount_1 = row_1[col1]
            if idx_1 in matched_df1_indices:
                continue

            for comb in combinations(not_conc_2_sorted.iterrows(), r):
                comb_indices = tuple(idx for idx, _ in comb)
                comb_sum = sum(row[col2] for _, row in comb)

                if comb_sum == amount_1:
                    matched_df1_indices.append(idx_1)
                    matched_df2_indices.extend(comb_indices)
                    not_conc_2_sorted = not_conc_2_sorted.drop(list(comb_indices))
                    break

    matched_df1 = og_df1.loc[matched_df1_indices]
    unmatched_df1 = og_df1.drop(matched_df1_indices)

    matched_df2 = og_df2.loc[matched_df2_indices]
    unmatched_df2 = og_df2.drop(matched_df2_indices)

    return matched_df1, matched_df2, unmatched_df1, unmatched_df2

def convert_df_to_csv_bytes(df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

# Streamlit app
st.title('Bank and Accounting Reconciliation')

uploaded_file_1 = st.file_uploader("Upload the first file", type=["csv", "xlsx", "xls"])
uploaded_file_2 = st.file_uploader("Upload the second file", type=["csv", "xlsx", "xls"])

if uploaded_file_1 and uploaded_file_2:
    header_1 = st.number_input('In which row is the header for the first file:', min_value=1) - 1
    header_2 = st.number_input('In which row is the header for the second file:', min_value=1) - 1

    df1 = read_file(uploaded_file_1, header_row=header_1)
    df2 = read_file(uploaded_file_2, header_row=header_2)

    file_type_1 = st.selectbox("Select the type for the first file:", ['Accounting', 'Bank'])
    file_type_2 = st.selectbox("Select the type for the second file:", ['Accounting', 'Bank'])

    try:
        if file_type_1 == 'Bank':
            date_col_1, amount_col_1 = prompt_columns(df1, 'Bank')
            processed_1 = process_bank(df1, date_col_1, amount_col_1)
        else:
            date_col_1, debit_col_1, credit_col_1 = prompt_columns(df1, 'Accounting')
            processed_1 = process_ledger(df1, date_col_1, debit_col_1, credit_col_1)

        if file_type_2 == 'Bank':
            date_col_2, amount_col_2 = prompt_columns(df2, 'Bank')
            processed_2 = process_bank(df2, date_col_2, amount_col_2)
        else:
            date_col_2, debit_col_2, credit_col_2 = prompt_columns(df2, 'Accounting')
            processed_2 = process_ledger(df2, date_col_2, debit_col_2, credit_col_2)

        not_conc_1, not_conc_2, conc_1, conc_2 = crossing(processed_1, processed_2)

        matched_df1, matched_df2, unmatched_df1, unmatched_df2 = find_matches_efficiently(
            df1, df2, not_conc_1, not_conc_2, conc_1, conc_2, 
            col1='Amount_Accounting' if file_type_1 == 'Accounting' else 'Amount_Bank',
            col2='Amount_Accounting' if file_type_2 == 'Accounting' else 'Amount_Bank'
        )

        st.write(f"Initial Reconciliation: {len(conc_1)} transactions reconciled.")

        st.write("Reconciled transactions from the first file:")
        st.dataframe(matched_df1)

        st.write("Reconciled transactions from the second file:")
        st.dataframe(matched_df2)

        st.write("Unreconciled transactions from the first file:")
        st.dataframe(unmatched_df1)

        st.write("Unreconciled transactions from the second file:")
        st.dataframe(unmatched_df2)

        st.write("Download results:")

        st.download_button(
            label="Download Reconciled Transactions - First File",
            data=convert_df_to_csv_bytes(matched_df1),
            file_name='reconciled_file_1.csv',
            mime='text/csv',
        )

        st.download_button(
            label="Download Reconciled Transactions - Second File",
            data=convert_df_to_csv_bytes(matched_df2),
            file_name='reconciled_file_2.csv',
            mime='text/csv',
        )

        st.download_button(
            label="Download Unreconciled Transactions - First File",
            data=convert_df_to_csv_bytes(unmatched_df1),
            file_name='unreconciled_file_1.csv',
            mime='text/csv',
        )

        st.download_button(
            label="Download Unreconciled Transactions - Second File",
            data=convert_df_to_csv_bytes(unmatched_df2),
            file_name='unreconciled_file_2.csv',
            mime='text/csv',
        )
    except KeyError:
        st.write('Please indicate the correct amount column.')
