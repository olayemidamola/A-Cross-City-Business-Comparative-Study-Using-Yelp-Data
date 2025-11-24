import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


ratings_df = review[['user_id', 'business_id', 'stars']].copy()


ratings_df['user_idx'] = ratings_df['user_id'].astype('category').cat.codes
ratings_df['business_idx'] = ratings_df['business_id'].astype('category').cat.codes

n_users = ratings_df['user_idx'].nunique()
n_items = ratings_df['business_idx'].nunique()


ratings_matrix = csr_matrix(
    (ratings_df['stars'], (ratings_df['user_idx'], ratings_df['business_idx'])),
    shape=(n_users, n_items)
)

train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_matrix = csr_matrix(
    (train_df['stars'], (train_df['user_idx'], train_df['business_idx'])),
    shape=(n_users, n_items)
)

# Train TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(train_matrix)

# Predict
pred_matrix = svd.transform(train_matrix) @ svd.components_

# RMSE
y_true, y_pred = [], []
for _, row in test_df.iterrows():
    u = row['user_idx']
    b = row['business_idx']
    y_true.append(row['stars'])
    y_pred.append(pred_matrix[u, b])

rmse_score = sqrt(mean_squared_error(y_true, y_pred))
print(f"Approximate RMSE: {rmse_score:.4f}")


export_df = ratings_df[['user_id', 'business_id', 'user_idx', 'business_idx', 'stars']].copy()
export_df.rename(columns={'stars': 'actual_stars'}, inplace=True)

export_df['predicted_stars'] = export_df.apply(
    lambda row: pred_matrix[row['user_idx'], row['business_idx']],
    axis=1
)

export_df = export_df.merge(user[['user_id', 'name']], on='user_id', how='left')
export_df = export_df.merge(business[['business_id', 'name']], on='business_id', how='left', suffixes=('_user', '_business'))

export_df.drop(columns=['user_idx', 'business_idx'], inplace=True)

export_df.to_csv('recommendation_predictions_sparse.csv', index=False)
print("✅ Export completed: recommendation_predictions_sparse.csv")
