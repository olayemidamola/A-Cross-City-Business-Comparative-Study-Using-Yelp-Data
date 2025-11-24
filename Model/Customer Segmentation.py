from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Data Cleaning & Preparation
# Business data
business = business[['business_id', 'name', 'city', 'stars', 'review_count', 'categories']]
business.dropna(subset=['categories'], inplace=True)
business = business[business['review_count'] > 5]

# Review data
review = review[['review_id', 'user_id', 'business_id', 'stars', 'text', 'date']]
review.drop_duplicates(subset=['review_id'], inplace=True)
review['text'] = review['text'].astype(str).str.strip()
review['date'] = pd.to_datetime(review['date'])

# User data
user = user[['user_id', 'name', 'review_count', 'average_stars']]

# Merge datasets
segmentation_df = review.merge(user, on='user_id').merge(business, on='business_id')

# Customer Segmentation (KMeans Clustering)
X = segmentation_df[['stars_x', 'average_stars', 'review_count_y']]
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
segmentation_df['cluster'] = kmeans.fit_predict(X_scaled)

# User Behavior Summary
user_behavior_df = segmentation_df.groupby('user_id').agg(
    name=('name_x', 'first'),
    cluster=('cluster', 'first'),
    total_reviews=('review_count_y', 'sum'),
    average_rating_given=('stars_x', 'mean'),
    average_stars_user=('average_stars', 'first'),
    last_active=('date', 'max')
).reset_index()

# Cluster Summary
cluster_summary_df = segmentation_df.groupby('cluster').agg(
    num_users=('user_id', 'nunique'),
    avg_review_count=('review_count_y', 'mean'),
    avg_rating_given=('stars_x', 'mean'),
    avg_user_stars=('average_stars', 'mean')
).reset_index()

# Top 5 Businesses per Cluster
top_n = 5
top_business_df = (
    segmentation_df.groupby(['cluster', 'business_id', 'name_y'])
    .agg(
        avg_rating=('stars_x', 'mean'),
        total_reviews=('review_id', 'count')
    )
    .reset_index()
    .sort_values(['cluster', 'total_reviews', 'avg_rating'], ascending=[True, False, False])
)
top_business_per_cluster = top_business_df.groupby('cluster').head(top_n)

# Rating Trends Over Time per Cluster
segmentation_df['year_month'] = segmentation_df['date'].dt.to_period('M')
rating_trends_df = segmentation_df.groupby(['cluster', 'year_month']).agg(
    avg_rating=('stars_x', 'mean'),
    num_reviews=('review_id', 'count')
).reset_index()
rating_trends_df['year_month'] = rating_trends_df['year_month'].astype(str)

# Export Everything into One Excel File
output_path = "customer_segmentation_report.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    segmentation_df.to_excel(writer, index=False, sheet_name='Full Segmentation')
    user_behavior_df.to_excel(writer, index=False, sheet_name='User Behavior')
    cluster_summary_df.to_excel(writer, index=False, sheet_name='Cluster Summary')
    top_business_per_cluster.to_excel(writer, index=False, sheet_name='Top Businesses')
    rating_trends_df.to_excel(writer, index=False, sheet_name='Rating Trends')

print("✅ Export completed: customer_segmentation_report.xlsx (5 sheets)")
