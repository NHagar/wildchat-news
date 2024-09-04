# %%
import duckdb
import numpy as np

# %%
con = duckdb.connect(":memory:")

# %%
counts = con.execute("SELECT classification, COUNT(*) AS records FROM '../data/batches_from_openai/all_data.parquet' GROUP BY 1").fetchdf()
# %%
total_records = counts['records'].sum()
records_classified_as_1 = counts[counts['classification'] == 1]['records'].values[0]

proportion = records_classified_as_1 / total_records
standard_error = np.sqrt((proportion * (1 - proportion)) / total_records)
margin_of_error = 1.96 * standard_error

lower_bound = proportion - margin_of_error
upper_bound = proportion + margin_of_error

print(f"Proportion of records classified as 1: {proportion:.4f}")
print(f"95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})")
