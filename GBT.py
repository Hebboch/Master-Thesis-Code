import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

"""
Purpose: Determine min-max demand intervals and corresponding standard deviation
Result:  pandas.DataFrame with columns: Area, Category, Weekday, Orders, Date
"""

def load_clean_rows(csv_path: str):
    """
    Input:  csv_path (str)
    Output: pandas.DataFrame with columns ['Area','Category','Weekday','Orders','Date']
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Exclude 'Desserts' (keep Burgers, Sides, Drinks)
    df = df[df['Category'] != 'Desserts'].copy()

    # Convert Date to weekday name
    df['Weekday'] = pd.to_datetime(df['Date']).dt.day_name()

    # Drop unused columns
    df = df.drop(columns=['Item', 'Description', 'Type of Order'])

    return df


def make_pipeline():
    """
    Input:  None
    Output: sklearn Pipeline ready for fit/predict on (Area, Category, Weekday)
    """
    # Preprocessor
    pre = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False),['Area', 'Category', 'Weekday'])],remainder='drop')

    # Regressor
    gbt = HistGradientBoostingRegressor(
        ## Settings per cited configuration
        loss='squared_error',   ### Gaussian
        learning_rate=0.01,     ### Learning rate
        max_depth=4,            ### Maximum tree depth
        max_iter=250,           ### Number of boosting iterations
        random_state=4          ### Seed for reproducibility
    )

    return Pipeline(steps=[('pre', pre), ('reg', gbt)])


def fit_on_rows(df_rows: pd.DataFrame):
    """
    Input:  df_rows (DataFrame) with columns ['Area','Category','Weekday','Orders']
    Output: Trained sklearn Pipeline
    """
    # Features
    X = df_rows[['Area', 'Category', 'Weekday']]
    y = df_rows['Orders']

    # Fit model
    model = make_pipeline()
    model.fit(X, y)

    return model


def prediction_pivot_from_rows(model: Pipeline, df_rows: pd.DataFrame):
    """
    Input:  model (Pipeline) fitted; df_rows with ['Area','Category','Weekday']
    Output: DataFrame pivot indexed by (Area, Category), columns = weekdays
    """
    # Predict
    X = df_rows[['Area', 'Category', 'Weekday']]
    tmp = df_rows.copy()
    tmp['Pred'] = model.predict(X)

    # Aggregate per (Area, Category, Weekday)
    agg_pred = tmp.groupby(['Area', 'Category', 'Weekday'], as_index=False)['Pred'].sum()

    # Create fancy table
    pivot = agg_pred.pivot_table(index=['Area', 'Category'], columns='Weekday', values='Pred').round().astype(int)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot[weekday_order]

    return pivot


def random_day_minmax_over_areas(pivot: pd.DataFrame, seed: int = 14):
    """
    Input:  pivot (DataFrame) from prediction_pivot_from_rows; seed (int) for reproducibility
    Output: DataFrame with index=Category and columns ['min','max','std']
    """
    # Choose weekday randomly
    rng = np.random.default_rng(seed)
    day = rng.choice(pivot.columns)

    ## Min/Max/Std per Category across all Areas for selected day
    finding = (pivot[[day]].rename(columns={day: 'Orders'}).reset_index().groupby('Category')['Orders'].agg(min='min', max='max', std='std')).round(0)

    return finding