from datetime import datetime
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import xgboost as xgb


class DelayModel:
    """
    Flight delay prediction model based on XGBoost.

    This class handles:
        - feature engineering
        - preprocessing
        - model training
        - prediction

    The model predicts whether a flight will be delayed
    by more than 15 minutes.
    """

    TOP_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    def __init__(self):
        """
        Initialize DelayModel.

        Attributes:
            _model:
                Trained XGBoost classifier.
                Initially set to None until fit() is called.
        """
        self.model = None
        self.data = None

    def get_period_day(self, date: str) -> str:
        """
        Categorize a flight according to the time of day.

        Args:
            date:
                Flight date in format '%Y-%m-%d %H:%M:%S'.

        Returns:
            str:
                One of:
                    - 'mañana'
                    - 'tarde'
                    - 'noche'
        """

        date_time = datetime.strptime(
            date,
            '%Y-%m-%d %H:%M:%S'
        ).time()

        morning_min = datetime.strptime(
            "05:00",
            '%H:%M'
        ).time()

        morning_max = datetime.strptime(
            "11:59",
            '%H:%M'
        ).time()

        afternoon_min = datetime.strptime(
            "12:00",
            '%H:%M'
        ).time()

        afternoon_max = datetime.strptime(
            "18:59",
            '%H:%M'
        ).time()

        if morning_min <= date_time <= morning_max:
            return 'mañana'

        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'

        return 'noche'

    def is_high_season(self, fecha: str) -> int:
        """
        Determine whether a flight date belongs to
        a high-season period.

        High-season ranges:
            - Dec 15 to Dec 31
            - Jan 1 to Mar 3
            - Jul 15 to Jul 31
            - Sep 11 to Sep 30

        Args:
            fecha:
                Flight date in format '%Y-%m-%d %H:%M:%S'.

        Returns:
            int:
                1 if the date is in high season,
                otherwise 0.
        """

        fecha_año = int(fecha.split('-')[0])

        fecha = datetime.strptime(
            fecha,
            '%Y-%m-%d %H:%M:%S'
        )

        ranges = [
            (
                datetime.strptime(
                    '15-Dec',
                    '%d-%b'
                ).replace(year=fecha_año),

                datetime.strptime(
                    '31-Dec',
                    '%d-%b'
                ).replace(year=fecha_año)
            ),

            (
                datetime.strptime(
                    '1-Jan',
                    '%d-%b'
                ).replace(year=fecha_año),

                datetime.strptime(
                    '3-Mar',
                    '%d-%b'
                ).replace(year=fecha_año)
            ),

            (
                datetime.strptime(
                    '15-Jul',
                    '%d-%b'
                ).replace(year=fecha_año),

                datetime.strptime(
                    '31-Jul',
                    '%d-%b'
                ).replace(year=fecha_año)
            ),

            (
                datetime.strptime(
                    '11-Sep',
                    '%d-%b'
                ).replace(year=fecha_año),

                datetime.strptime(
                    '30-Sep',
                    '%d-%b'
                ).replace(year=fecha_año)
            )
        ]

        for start, end in ranges:

            if start <= fecha <= end:
                return 1

        return 0

    def get_min_diff(self, row: pd.Series) -> float:
        """
        Compute the delay in minutes between the
        scheduled and actual flight timestamps.

        Args:
            row:
                DataFrame row containing:
                    - Fecha-I
                    - Fecha-O

        Returns:
            float:
                Difference in minutes between
                Fecha-O and Fecha-I.
        """

        fecha_o = datetime.strptime(
            row['Fecha-O'],
            '%Y-%m-%d %H:%M:%S'
        )

        fecha_i = datetime.strptime(
            row['Fecha-I'],
            '%Y-%m-%d %H:%M:%S'
        )

        return (
            (fecha_o - fecha_i).total_seconds()
        ) / 60

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        pd.DataFrame
    ]:
        """
        Transform raw flight data into model-ready features.

        The preprocessing pipeline performs:
            - time-of-day extraction
            - high-season detection
            - delay calculation
            - one-hot encoding
            - feature selection

        Args:
            data:
                Raw flight dataset.

            target_column:
                Name of the target column to create.
                If provided, the target variable is returned.

        Returns:
            Union[
                Tuple[pd.DataFrame, pd.DataFrame],
                pd.DataFrame
            ]

            If target_column is provided:
                - features dataframe
                - target series

            Otherwise:
                - features dataframe only
        """

        data = data.copy()

        # Feature engineering
        data['period_day'] = data['Fecha-I'].apply(
            self.get_period_day
        )

        data['high_season'] = data['Fecha-I'].apply(
            self.is_high_season
        )

        data['min_diff'] = data.apply(
            self.get_min_diff,
            axis=1
        )

        # Target generation
        if target_column is not None:

            threshold_in_minutes = 15

            data[target_column] = np.where(
                data['min_diff'] > threshold_in_minutes,
                1,
                0
            )

        # One-hot encoding
        features = pd.concat([
            pd.get_dummies(
                data['OPERA'],
                prefix='OPERA'
            ),

            pd.get_dummies(
                data['TIPOVUELO'],
                prefix='TIPOVUELO'
            ),

            pd.get_dummies(
                data['MES'],
                prefix='MES'
            )

        ], axis=1)

        # Ensure all required columns exist
        for col in self.TOP_FEATURES:

            if col not in features.columns:
                features[col] = 0

        # Keep selected features only
        features = features[self.TOP_FEATURES]

        if target_column is not None:
            # target = data[target_column]
            target = data[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Train the XGBoost classifier.

        The model uses class balancing to compensate
        for delay class imbalance.

        Args:
            features:
                Preprocessed training features.

            target:
                Training target labels.
        """
        target_series = target.iloc[:, 0]

        n_y0 = (target_series == 0).sum()
        n_y1 = (target_series == 1).sum()

        scale = n_y0 / n_y1


        self.model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale
        )

        self.model.fit(
            features,
            target
        )

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict whether flights will be delayed.

        Args:
            features:
                Preprocessed feature dataframe.

        Returns:
            List[int]:
                Binary predictions where:
                    - 1 = delayed flight
                    - 0 = non-delayed flight
        """

        predictions = self.model.predict(features)

        return predictions.astype(int).tolist()