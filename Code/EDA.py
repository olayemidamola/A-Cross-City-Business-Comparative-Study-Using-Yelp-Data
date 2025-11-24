# Business
df = business.copy() 
print(df.shape)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numeric Columns:", num_cols)
print("Categorical Columns:", cat_cols)

#### Univariate Analysis
# Numeric variables
df[num_cols].describe().T

for col in num_cols:
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram: {col}")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot: {col}")

    plt.tight_layout()
    plt.show()

# Categorical variables
cat_cols = [
    col for col in df.columns 
    if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, str)).all()
]

for col in cat_cols:
    top_vals = df[col].value_counts().head(15)

    plt.figure(figsize=(8,4))
    sns.barplot(x=top_vals.values, y=top_vals.index)
    plt.title(f"Top categories: {col}")
    plt.tight_layout()
    plt.show()

#### Bivariate Analysis
# Number vs. Number
corr = df[num_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

for col in num_cols:
    if col != "stars":
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df["stars"], alpha=0.3)
        plt.title(f"{col} vs stars")
        plt.show()

# Numerical vs. Category
top_states = df["state"].value_counts().head(15).index

plt.figure(figsize=(12,5))
sns.boxplot(data=df[df["state"].isin(top_states)], x="state", y="stars")
plt.xticks(rotation=45)
plt.title("Stars by State (Top 15)")
plt.show()

#### Multivariate Analysis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X = df[num_cols]

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame({
    "PC1": pcs[:,0],
    "PC2": pcs[:,1],
    "stars": df["stars"]
})

print("Explained Variance:", pca.explained_variance_ratio_)

plt.figure(figsize=(7,6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="stars", palette="viridis", alpha=0.6)
plt.title("PCA Projection Colored by Stars")
plt.show()

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df_pca["cluster"] = labels

plt.figure(figsize=(7,6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="cluster", palette="Set2", alpha=0.7)
plt.title("KMeans Clusters on PCA Components")
plt.show()

#### Advance Analysis
# Distribution of Business Star Ratings
sns.histplot(business['stars'], bins=20, kde=True, palette='magma') # 显示出 核密度估计曲线（Kernel Density Estimate）
plt.title('Distribution of Business Star Ratings')
plt.xlabel('Star Rating')
plt.ylabel('Frequency')
plt.show()

# Top 20 Cities by Number of Businesses
top20_city = business['city'].value_counts(ascending=False).head(20)
sns.barplot(x=top20_city.values, y=top20_city.index, palette='magma')
plt.title('Top 20 Cities by Number of Businesses')
plt.xlabel('Number of Businesses')
plt.ylabel('City')
plt.show()

# Review Count vs. Star Ratings
sns.scatterplot(x='review_count', y='stars', data=business, palette='magma')
plt.title('Review Count vs. Star Ratings')
plt.xlabel('Review Count')
plt.ylabel('Star Ratings')
plt.show()

# Number of Businesses by Main Category
sns.barplot(x=business['main_cat'].value_counts().values, y=business['main_cat'].value_counts().index, palette='magma')
plt.title('Number of Businesses by Main Category')
plt.xlabel('Number of Businesses')
plt.ylabel('Main Category')
plt.show()

# Average Star Ratings by Main Category
sns.barplot(x='main_cat', y='stars', data=business, palette='magma')
plt.title('Average Star Ratings by Main Category')
plt.xlabel('Main Category')
plt.ylabel('Average Star Ratings')
plt.xticks(rotation=45)
plt.show()

numeric_cols = business.select_dtypes(include=['int64', 'float64']).columns
business_numeric = business[numeric_cols]
corr_matrix = business_numeric.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Matrix of Business Numeric Columns")
plt.show()

# Checkin
#### Univariate Analysis
def univariate_analysis(checkin):
    print("===== Univariate Analysis =====")

    numeric_cols = ["year", "month", "day_of_week", "hour"]

    # Summary statistics
    summary = checkin[numeric_cols].describe().T
    summary["missing"] = checkin[numeric_cols].isna().sum()
    summary["unique"] = checkin[numeric_cols].nunique()
    print(summary)

    # Histograms
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(checkin[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


univariate_analysis(checkin)

#### Bivariate Analysis
def bivariate_analysis(checkin):
    print("===== Bivariate Analysis =====")

    numeric_cols = ["year", "month", "day_of_week", "hour"]

    # Correlation matrix
    plt.figure(figsize=(6, 5))
    corr = checkin[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Scatter plots: pairs of numeric variables
    sns.pairplot(checkin[numeric_cols], plot_kws={'alpha':0.4, 's':20})
    plt.suptitle("Pairwise Relationships Between Time Features", y=1.02)
    plt.show()

    # Bivariate categorical-like analysis
    # Most active hours by day of week
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="day_of_week", y="hour", data=checkin)
    plt.title("Check-in Hour by Day of Week")
    plt.xlabel("Day of Week (0=Mon)")
    plt.ylabel("Hour of Day")
    plt.tight_layout()
    plt.show()

bivariate_analysis(checkin)

#### Multivariate Analysis
def multivariate_analysis(checkin):
    print("===== Multivariate Analysis =====")

    numeric_cols = ["year", "month", "day_of_week", "hour"]

    # Scaling data
    X = checkin[numeric_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # PCA scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_components[:,0], pca_components[:,1], alpha=0.4, s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Check-in Time Features")
    plt.tight_layout()
    plt.show()

    # KMeans clustering on PCA results
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(pca_components)

    plt.figure(figsize=(6, 5))
    plt.scatter(pca_components[:,0], pca_components[:,1], c=labels, cmap="viridis", s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans Clusters (k=3) on PCA Features")
    plt.tight_layout()
    plt.show()


multivariate_analysis(checkin)

#### Advance Analysis
# Check-in by Year
sns.countplot(x='year', data=checkin)
plt.title('Check-ins by Year')
plt.xlabel('Year')
plt.ylabel('Number of Check-ins')
plt.xticks(rotation=45)
plt.show()

# Check-in by Month
sns.countplot(x='month', data=checkin)
plt.title('Check-ins by Month')
plt.xlabel('Month')
plt.ylabel('Number of Check-ins')
plt.show()

# Rralationship between Weekday and Hour
pivot = checkin.pivot_table(index='day_of_week', columns='hour', values='business_id', aggfunc='count')
sns.heatmap(pivot, cmap='Blues')
plt.title('Check-in Heatmap by Weekday and Hour')
plt.xlabel('Hour')
plt.ylabel('Day of Week')
plt.show()

# Review Data
#### Univariate Analysis
# Numeric
review = review.head(1000)

# =============================
# 1. Select only numeric columns
# =============================
numeric_cols = review.select_dtypes(include=['int64', 'float64']).columns
print("Numeric Columns:", list(numeric_cols))

# =============================
# 2. Summary statistics
# =============================
print("\n===== SUMMARY STATISTICS =====")
print(review[numeric_cols].describe())

# =============================
# 3. Missing values in numeric columns
# =============================
print("\n===== MISSING VALUES =====")
print(review[numeric_cols].isnull().sum())

# =============================
# 4. Univariate Plots
# =============================
for col in numeric_cols:
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(review[col], kde=True)
    plt.title(f"Histogram of {col}")

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=review[col])
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

# Categorical
# 1. Select only categorical columns
# =============================
cat_cols = review.select_dtypes(include=['object', 'category']).columns
print("Categorical Columns:", list(cat_cols))

# =============================
# 2. Unique value counts
# =============================
print("\n===== UNIQUE VALUE COUNTS =====")
for col in cat_cols:
    print(f"\nColumn: {col}")
    print(review[col].nunique(), "unique values")
    print(review[col].value_counts().head(10))  # top 10 most frequent

# =============================
# 3. Frequency distribution tables
# =============================
print("\n===== FREQUENCY DISTRIBUTION TABLES (Top 10 values) =====")
for col in cat_cols:
    print(f"\nFrequency table for {col}:")
    print(review[col].value_counts().head(10))

# =============================
# 4. Univariate Bar Charts
# =============================
for col in cat_cols:
    plt.figure(figsize=(10, 5))

    # show top 20 categories only (to avoid messy plots)
    top_values = review[col].value_counts().head(20)

    sns.barplot(x=top_values.index, y=top_values.values)
    plt.title(f"Top 20 Categories in {col}")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(col)
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


from wordcloud import WordCloud, STOPWORDS

#check the most common words used by customers

# =============================
# 1. Prepare text data
# =============================
text_data = " ".join(review['text'].astype(str).tolist())

# Add default stopwords + custom ones
stopwords = set(STOPWORDS)
stopwords.update(["the", "and", "to", "for", "is", "it", "was"])  # optional

# =============================
# 2. Generate WordCloud
# =============================
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='white',
    stopwords=stopwords,
    max_words=200
).generate(text_data)

# =============================
# 3. Display WordCloud
# =============================
plt.figure(figsize=(18, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Yelp Review Text", fontsize=18)
plt.show()

#### Bivariate Analysis
# =============================
# 1. Select numeric columns only
# =============================
num_cols = review.select_dtypes(include=['int64', 'float64']).columns
print("Numeric Columns:", list(num_cols))

# =============================
# 2. Correlation Matrix
# =============================
corr_matrix = review[num_cols].corr()

print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Columns - Yelp Reviews)")
plt.show()

# =============================
# 3. Pairplot (numeric vs numeric)
# =============================
sns.pairplot(review[num_cols], diag_kind='hist')
plt.suptitle("Pairwise Scatterplots (Numeric Columns)", y=1.02)
plt.show()

# =============================
# 4. Individual Scatterplots with Regression Lines
# =============================
plt.figure(figsize=(12, 8))
num_list = list(num_cols)

for i in range(len(num_list)):
    for j in range(i + 1, len(num_list)):
        col1 = num_list[i]
        col2 = num_list[j]

        plt.figure(figsize=(6,4))
        sns.regplot(x=review[col1], y=review[col2], scatter_kws={'alpha':0.3})
        plt.title(f"{col1} vs {col2} (with Regression Line)")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.show()

# Categorical 

# =========================================
# LIMIT DATASET TO FIRST 1,000 ROWS
# =========================================
review = review.head(1000)

# =========================================
# 1. Identify categorical and numeric columns
# =========================================
cat_cols = review.select_dtypes(include=['object', 'category']).columns
num_cols = review.select_dtypes(include=['int64', 'float64']).columns

print("Categorical Columns:", list(cat_cols))
print("Numeric Columns:", list(num_cols))

# =========================================
# 2. Bivariate: Categorical vs Numeric
# =========================================
for cat in cat_cols:
    for num in num_cols:
        # Boxplot
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=review[cat], y=review[num])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{num} by {cat}")
        plt.tight_layout()
        plt.show()

        # Optional: bar chart of mean numeric value per category
        plt.figure(figsize=(10, 5))
        review.groupby(cat)[num].mean().sort_values(ascending=False).head(20).plot(kind='bar')
        plt.title(f"Mean {num} per {cat} (Top 20)")
        plt.ylabel(f"Mean {num}")
        plt.tight_layout()
        plt.show()

#### Multivariate Analysis
# =========================================
# LIMIT DATASET TO FIRST 1,000 ROWS
# =========================================
review = review.head(1000)

# =========================================
# Identify categorical and numeric columns
# =========================================
cat_cols = review.select_dtypes(include=['object', 'category']).columns
num_cols = review.select_dtypes(include=['int64', 'float64']).columns

print("Categorical Columns:", list(cat_cols))
print("Numeric Columns:", list(num_cols))

# =========================================
# 1. Numeric vs Numeric: Correlation Heatmap
# =========================================
plt.figure(figsize=(8, 6))
corr = review[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Numeric Columns")
plt.tight_layout()
plt.show()


# =========================================
# 2. Numeric vs Categorical: Boxplots / FacetGrid
# =========================================
for num in num_cols:
    for cat in cat_cols:
        plt.figure(figsize=(12, 5))
        sns.boxplot(x=review[cat], y=review[num])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{num} by {cat}")
        plt.tight_layout()
        plt.show()

#### Advance Analysis
# Basic stat
print("Basic Statistics:")
print(review.describe(include='all'))

# --- 2 Star Rating Distribution ---
plt.figure(figsize=(8,5))
sns.countplot(x='stars', data=review, palette='viridis')
plt.title('Distribution of Yelp Ratings')
plt.xlabel('Stars')
plt.ylabel('Count')
plt.show()

# --- 3️ Review Text Length Analysis ---
review['text_length'] = review['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8,5))
sns.histplot(review['text_length'], bins=50, kde=True)
plt.title('Distribution of Review Text Length')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.show()

# --- 4️Most Active Users ---
top_users = review['user_id'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_users.index, y=top_users.values, palette='magma')
plt.title('Top 10 Most Active Users')
plt.xlabel('User ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

# --- 5️ Most Reviewed Businesses ---
top_businesses = review['business_id'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_businesses.index, y=top_businesses.values, palette='plasma')
plt.title('Top 10 Most Reviewed Businesses')
plt.xlabel('Business ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

# --- 6️ Average Rating per Business ---
avg_rating_business = review.groupby('business_id')['stars'].mean().sort_values(ascending=False).head(10)
print("Top 10 Businesses by Average Rating:\n", avg_rating_business)

# --- 7️ Word Cloud of Review Text ---
all_text = ' '.join(review['text'].astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)

plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Yelp Reviews')
plt.show()

# Photos Data
# --- 1️ Basic Statistics ---
print("Basic Statistics:")
print(photos.describe(include='all'))

# --- 2️ Number of Photos per Business ---
photos_per_business = photos['business_id'].value_counts()
plt.figure(figsize=(10,5))
sns.histplot(photos_per_business, bins=50, kde=True)
plt.title('Distribution of Number of Photos per Business')
plt.xlabel('Number of Photos')
plt.ylabel('Count of Businesses')
plt.show()

# --- 3️ Distribution of Labels / Categories ---
plt.figure(figsize=(12,6))
sns.countplot(y='main_cat', data=photos, order=photos['main_cat'].value_counts().index, palette='viridis')
plt.title('Distribution of Main Categories in Photos')
plt.xlabel('Count')
plt.ylabel('Main Category')
plt.show()

photos['caption_length'] = photos['caption'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8,5))
sns.histplot(photos['caption_length'], bins=50, kde=True)
plt.title('Distribution of Caption Text Length')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.show()

# --- 5️ Most Frequent Captions ---
top_captions = photos['caption'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_captions.values, y=top_captions.index, palette='magma')
plt.title('Top 10 Most Frequent Captions')
plt.xlabel('Count')
plt.ylabel('Caption')
plt.show()

# --- 6️ Word Cloud of Photo Captions ---
all_captions = ' '.join(photos['caption'].astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_captions)

plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Photo Captions')
plt.show()

# User
user_df = user.head(1000).copy()  

num_cols_user = user_df.select_dtypes(include=['int64','float64']).columns
cat_cols_user = user_df.select_dtypes(include=['object']).columns

#### Univariate Analysis
# Numeric
for col in num_cols_user:
    plt.figure(figsize=(8,4))
    sns.histplot(user_df[col], kde=True, bins=40, color='steelblue')
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Categorical
for col in cat_cols_user:
    plt.figure(figsize=(10,4))
    user_df[col].value_counts().head(20).plot(kind='bar', color='purple')
    plt.title(f"Top 20 Most Frequent Categories in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

#### Bivariate Analysis
# Numeric vs Numeric
sns.pairplot(user_df[num_cols_user], height=2)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(user_df[num_cols_user].corr(), annot=False, cmap="Blues")
plt.title("Correlation Heatmap — User Numeric Features", fontsize=15)
plt.show()

#### Advance Analysis
# Distribution of review counts
plt.figure(figsize=(8,4))
sns.histplot(user_clean['review_count'], bins=50, kde=True)
plt.title('Distribution of Review Counts per User')
plt.xlabel('Review Count')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap for numeric columns
plt.figure(figsize=(8,5))
sns.heatmap(user_clean.corr(numeric_only=True), annot=True, cmap='Blues')
plt.title('Correlation Heatmap - User Dataset')
plt.show()

# Top users by review count
top_users = user_clean.nlargest(10, 'review_count')[['name','review_count']]
plt.figure(figsize=(8,4))
sns.barplot(data=top_users, x='review_count', y='name', palette='magma')
plt.title('Top 10 Users by Review Count')
plt.xlabel('Review Count')
plt.ylabel('User')
plt.show()

# Tip
tip_df = tip.head(1000).copy()   #loaded tip dataset

#### Univariate Analysis
# Numeric 
num_cols_tip = ['compliment_count']

for col in num_cols_tip:
    plt.figure(figsize=(8,4))
    sns.histplot(tip_df[col], kde=True, bins=40, color='seagreen')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Categorical
cat_cols_tip = ['user_id','business_id','text']

for col in cat_cols_tip:
    plt.figure(figsize=(10,4))
    tip_df[col].value_counts().head(20).plot(kind='bar', color='darkred')
    plt.title(f"Top 20 in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Time Series
tip_df['year'] = tip_df['date'].dt.year

plt.figure(figsize=(8,4))
sns.countplot(x='year', data=tip_df, color='teal')
plt.title("Number of Tips by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

#### Bivariate Analysis
# Numeric vs Numeric
tip_df['month'] = tip_df['date'].dt.month

plt.figure(figsize=(8,4))
sns.scatterplot(x='month', y='compliment_count', data=tip_df)
plt.title("Compliment Count by Month")
plt.xlabel("Month")
plt.ylabel("Compliment Count")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(tip_df[['compliment_count','month']].corr(), annot=True, cmap="Purples")
plt.title("Correlation Heatmap — Tip Dataset")
plt.show()

# Categorical vs Categorical 
from itertools import combinations

for c1, c2 in combinations(cat_cols_tip, 2):
    top1 = tip_df[c1].value_counts().head(20).index
    top2 = tip_df[c2].value_counts().head(20).index

    ct = pd.crosstab(tip_df[tip_df[c1].isin(top1)][c1],
                     tip_df[tip_df[c2].isin(top2)][c2])

    plt.figure(figsize=(10,6))
    sns.heatmap(ct, cmap="coolwarm")
    plt.title(f"Heatmap: {c1} vs {c2}")
    plt.show()

#### Advance Analysis
# Tip text length distribution
tip['text_length'] = tip['text'].str.len()

plt.figure(figsize=(8,4))
sns.histplot(tip['text_length'], bins=50, kde=True)
plt.title('Distribution of Tip Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Top users by number of tips
top_tippers = tip['user_id'].value_counts().head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=top_tippers.values, y=top_tippers.index, palette='viridis')
plt.title('Top 10 Users by Number of Tips')
plt.xlabel('Number of Tips')
plt.ylabel('User ID')
plt.show()

# Temporal trend of tips
tip['year'] = tip['date'].dt.year
tips_per_year = tip.groupby('year')['text'].count()

plt.figure(figsize=(8,4))
tips_per_year.plot(kind='bar', color='teal')
plt.title('Number of Tips per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

