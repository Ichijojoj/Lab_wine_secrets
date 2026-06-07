from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import configparser
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from src.data_preprocessing import WineDataPreprocessor


class ModelTrainer:
    def __init__(self, config_path='configs/config.ini'):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.config.read(config_path)
        self.preprocessor = WineDataPreprocessor(config_path)

    def train(self):
        print("\nЗагрузка данных...")
        df = self.preprocessor.load_data()
        print(f"Загружено {len(df)} образцов")

        print("\nПредобработка...")
        X, y, feature_columns = self.preprocessor.preprocess_data(df)
        print(f" Признаков: {len(feature_columns)}")

        try:
            test_size = float(self.config['MODEL']['test_size'])
            random_state = int(self.config['MODEL']['random_state'])
            n_estimators = int(self.config['MODEL']['n_estimators'])
            max_depth = int(self.config['MODEL']['max_depth'])
            min_samples_split = int(self.config['MODEL']['min_samples_split'])
            min_samples_leaf = int(self.config['MODEL']['min_samples_leaf'])
            model_path = self.config['MODEL']['model_path']
            scaler_path = self.config['MODEL']['scaler_path']
            metrics_path = self.config['MODEL']['metrics_path']
        except KeyError as e:
            raise KeyError(f"Missing required parameter in configuration: {e}")

        X_train, X_test, y_train, y_test, scaler = self.preprocessor.split_and_scale(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        print(f" Train: {len(X_train)}, Test: {len(X_test)}")

        print("\n Обучение Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(X_train, y_train)
        print(" Модель обучена!")

        # Оценка
        print("\n Оценка модели...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f" ROC-AUC: {roc_auc:.4f}")
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))

        print("\n Сохранение модели...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_names': feature_columns,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
        joblib.dump(metrics, metrics_path)

        print(f" Модель сохранена: {model_path}")
        print(f" Scaler сохранён: {scaler_path}")
        print(f" Метрики сохранены: {metrics_path}")

        return model, metrics

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()