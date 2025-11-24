# Review Data
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

review['date'] = pd.to_datetime(review['date'])

#Text
review['text_length'] = review['text'].apply(len)
review['word_count'] = review['text'].apply(lambda x: len(str(x).split()))
review['avg_word_length'] = review['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)

#Sentiment
review['polarity'] = review['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
review['subjectivity'] = review['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
review['sentiment'] = review['polarity'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))

# Business Data
# Numerical Variables
# Popularity features (comprehensive rating + number of reviews)
business['popularity'] = np.log1p(business['review_count']) * business['stars']

# Check-in aggregation: total number of check-ins per merchant
checkin_count = checkin.groupby('business_id').size().reset_index(name='checkin_count')
business = business.merge(checkin_count, on='business_id', how='left')
business['checkin_count'] = business['checkin_count'].fillna(0)
business['checkin_log'] = np.log1p(business['checkin_count'])

# Categorical Variables
from sklearn.preprocessing import LabelEncoder
le_cat = LabelEncoder()
le_city = LabelEncoder()

business['main_cat_enc'] = le_cat.fit_transform(business['main_cat'])
business['city_enc'] = le_city.fit_transform(business['city'])

# Geographic clustering: cluster businesses into several areas based on longitude and latitude
from sklearn.cluster import KMeans

coords = business[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=20, random_state=42)
business['geo_cluster'] = kmeans.fit_predict(coords)

# TF-IDF vectorizes category information
# Building content similarity (content-based recommendation)
tfidf = TfidfVectorizer(max_features=200)

category_tfidf = tfidf.fit_transform(business['categories'])
category_tfidf_df = pd.DataFrame(category_tfidf.toarray(), columns=tfidf.get_feature_names_out())

business_tfidf = pd.concat([business.reset_index(drop=True), category_tfidf_df.reset_index(drop=True)], axis=1)

# Attributes
def parse_attributes(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}

attr_df = business_tfidf['attributes'].apply(parse_attributes).apply(pd.Series)
attr_df = attr_df.fillna(0)

for col in attr_df.columns:
    attr_df[col] = attr_df[col].astype(str).replace({'True':1, 'False':0})

business_attr = pd.concat([business_tfidf, attr_df.add_prefix('attr_')], axis=1)

# Time feature (business hours) extracts "business days" and "average daily business hours"
def hours_to_features(h):
    try:
        hours = ast.literal_eval(h)
        days = len(hours)
        total_hours = 0
        for k,v in hours.items():
            open_, close_ = v.split('-')
            total_hours += abs(int(close_.split(':')[0]) - int(open_.split(':')[0]))
        avg_hours = total_hours / days if days > 0 else 0
        return pd.Series({'open_days': days, 'avg_daily_hours': avg_hours}) # DateFrame to Series
    except:
        return pd.Series({'open_days': 0, 'avg_daily_hours': 0})

hours_features = business_attr['hours'].apply(hours_to_features)
business_feat = pd.concat([business_attr, hours_features], axis=1)

# Standardization and eventual integration
from sklearn.preprocessing import StandardScaler
num_features = ['stars','review_count','popularity','checkin_log','avg_daily_hours']
scaler = StandardScaler()
business_feat[num_features] = scaler.fit_transform(business_feat[num_features])

# Photos
#  Numerical Features
# ----------------------------

# Caption length (number of words)
photos['caption_length'] = photos['caption'].apply(lambda x: len(str(x).split()))

# Optional: log-transform for skewed features (e.g., if photos per business is added later)
photos_per_business = photos.groupby('business_id').size().reset_index(name='photo_count')
photos = photos.merge(photos_per_business, on='business_id', how='left')
photos['photos_count_log'] = np.log1p(photos['photo_count'])


#  Categorical Features
# ----------------------------

# Encode business_id and main category
le_business = LabelEncoder()
le_main_cat = LabelEncoder()

photos['business_id_enc'] = le_business.fit_transform(photos['business_id'])
photos['main_cat_enc'] = le_main_cat.fit_transform(photos['main_cat'])


#  Text Features
# ----------------------------

# TF-IDF vectorization for captions
tfidf_caption = TfidfVectorizer(max_features=100)  # Limit to top 100 words
caption_tfidf = tfidf_caption.fit_transform(photos['caption'])

caption_tfidf_df = pd.DataFrame(caption_tfidf.toarray(), columns=tfidf_caption.get_feature_names_out())

# Combine TF-IDF features with photo dataframe
photos_features = pd.concat([photos.reset_index(drop=True), caption_tfidf_df.reset_index(drop=True)], axis=1)

# Save feature-engineered photo dataset
# ----------------------------
photos_features.to_csv('photos_features.csv', index=False)

print("Feature engineering completed for the photo dataset!")

# Checkin Data
# Business-level Features
checkin_feat = (
    checkin.groupby('business_id')
    .agg(
        checkin_count=('checkin_time', 'count'),
        active_months=('month', pd.Series.nunique),
        active_hours=('hour', pd.Series.nunique),
        weekend_ratio=('day_of_week', lambda x: (x>=5).mean()),
        night_ratio=('hour', lambda x: ((x>=20) | (x<=6)).mean())
    )
    .reset_index()
)

business_feat = business.merge(checkin_feat, on='business_id', how='left', suffixes=('', '_checkin'))
business_feat['checkin_count'] = business_feat['checkin_count'].fillna(0)

business_feat.to_csv('business_checkin.csv', index=False)

print("Feature engineering completed for the business_checkin dataset!")

# User
# Feature: Average compliment score
compliment_cols = [col for col in user_clean.columns if 'compliment_' in col]
user_clean['total_compliments'] = user_clean[compliment_cols].sum(axis=1)

# Feature: Average influence score (weighted)
user_clean['influence_score'] = (
    user_clean['fans'] * 0.5 +
    user_clean['review_count'] * 0.3 +
    user_clean['useful'] * 0.2
)

# Feature: Elite user flag
user_clean['is_elite'] = np.where(user_clean['elite'] != 'None', 1, 0)

# Select relevant columns for modeling
user_features = user_clean[['user_id', 'review_count', 'fans', 'account_age',
                            'total_compliments', 'influence_score', 'is_elite']]
user_features.head()

# Tip
# Feature: Tip length (already computed)
tip['text_length'] = tip['text'].str.len()

# Feature: Average compliment count per user
tip_user_stats = tip.groupby('user_id').agg(
    avg_tip_length=('text_length','mean'),
    total_likes=('compliment_count','sum'),
    total_tips=('text','count')
).reset_index()

# Merge with user dataset (if needed)
merged_data = pd.merge(tip_user_stats, user_features, on='user_id', how='left')

merged_data.head()