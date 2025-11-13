"""
Sistema de Predicción de Contacto en Cobranzas
==============================================

Módulo principal que contiene todas las clases para entrenar, evaluar y simular
modelos de predicción de contacto en intentos de cobranza.

Autor: Sistema de ML para Cobranzas
Fecha: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

# Modelos lineales y discriminantes
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# SVM
from sklearn.svm import SVC, LinearSVC

# Naive Bayes y KNN
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

# Árboles y Ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier
)

# Redes Neuronales
from sklearn.neural_network import MLPClassifier

# Gradient Boosting avanzados
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Modelos específicos para desbalance (imblearn)
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    IMBLEARN_ENSEMBLE_AVAILABLE = True
except:
    IMBLEARN_ENSEMBLE_AVAILABLE = False

# Sampling (ya estaban)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Métricas
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, average_precision_score
)

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Clase para crear nuevas features mediante ingeniería de características.
    Maneja correctamente NaNs e infinitos.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.created_features = []
    
    def create_ratio_features(self, df):
        """
        Crea features de ratios entre gestiones.
        Ejemplos: tasa_efectividad, ratio_contacto_directo, etc.
        """
        if self.verbose:
            print("\n🔧 Creando features de RATIOS...")
        
        df_new = df.copy()
        created = []
        
        # Ratios de efectividad por periodo
        for period in ['3UM', '6UM', '9UM', '12UM']:
            # Ratio: Gestiones Efectivas / Totales
            col_efect = f'AVG_GestEfect_CallOut_{period}'
            col_total = f'AVG_GestTotales_CallOut_{period}'
            
            if col_efect in df.columns and col_total in df.columns:
                new_col = f'Ratio_Efectividad_{period}'
                df_new[new_col] = df[col_efect] / (df[col_total] + 1e-6)  # Evitar división por cero
                df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
                created.append(new_col)
            
            # Ratio: Contacto Directo / Efectivas
            col_directo = f'AVG_GestDirecto_CallOut_{period}'
            if col_directo in df.columns and col_efect in df.columns:
                new_col = f'Ratio_Directo_de_Efectivas_{period}'
                df_new[new_col] = df[col_directo] / (df[col_efect] + 1e-6)
                df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
                created.append(new_col)
            
            # Ratio: Sin Contacto / Totales
            col_sincontacto = f'AVG_GestSinContacto_CallOut_{period}'
            if col_sincontacto in df.columns and col_total in df.columns:
                new_col = f'Ratio_SinContacto_{period}'
                df_new[new_col] = df[col_sincontacto] / (df[col_total] + 1e-6)
                df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
                created.append(new_col)
        
        # Ratios de WhatsApp
        for period in ['3UM', '6UM', '9UM', '12UM']:
            col_wpp = f'AVG_GEST_WPP_OUT_{period}'
            col_total = f'AVG_GestTotales_CallOut_{period}'
            
            if col_wpp in df.columns and col_total in df.columns:
                new_col = f'Ratio_WhatsApp_Total_{period}'
                df_new[new_col] = df[col_wpp] / (df[col_total] + 1e-6)
                df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
                created.append(new_col)
        
        self.created_features.extend(created)
        
        if self.verbose:
            print(f"   ✓ Creadas {len(created)} features de ratios")
        
        return df_new
    
    def create_trend_features(self, df):
        """
        Crea features de tendencias temporales.
        Compara periodos recientes vs antiguos.
        """
        if self.verbose:
            print("\n🔧 Creando features de TENDENCIAS temporales...")
        
        df_new = df.copy()
        created = []
        
        # Tendencias de gestiones (3UM vs 12UM)
        for gest_type in ['GestTotales', 'GestEfect', 'GestDirecto', 'GestSinContacto']:
            col_recent = f'AVG_{gest_type}_CallOut_3UM'
            col_old = f'AVG_{gest_type}_CallOut_12UM'
            
            if col_recent in df.columns and col_old in df.columns:
                # Tendencia: (reciente - antiguo) / (antiguo + 1)
                new_col = f'Tendencia_{gest_type}_3vs12'
                df_new[new_col] = (df[col_recent] - df[col_old]) / (df[col_old] + 1)
                df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
                created.append(new_col)
        
        # Tendencias de WhatsApp
        col_wpp_recent = 'AVG_GEST_WPP_OUT_3UM'
        col_wpp_old = 'AVG_GEST_WPP_OUT_12UM'
        if col_wpp_recent in df.columns and col_wpp_old in df.columns:
            new_col = 'Tendencia_WhatsApp_3vs12'
            df_new[new_col] = (df[col_wpp_recent] - df[col_wpp_old]) / (df[col_wpp_old] + 1)
            df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
            created.append(new_col)
        
        self.created_features.extend(created)
        
        if self.verbose:
            print(f"   ✓ Creadas {len(created)} features de tendencias")
        
        return df_new
    
    def create_density_features(self, df):
        """
        Crea features de densidad (gestiones por unidad de tiempo/deuda).
        """
        if self.verbose:
            print("\n🔧 Creando features de DENSIDAD...")
        
        df_new = df.copy()
        created = []
        
        # Gestiones por día de mora
        if 'dias_mora_cartera' in df.columns:
            for period in ['3UM', '6UM', '9UM', '12UM']:
                col_gest = f'AVG_GestTotales_CallOut_{period}'
                if col_gest in df.columns:
                    new_col = f'Densidad_Gestiones_por_DiaMora_{period}'
                    df_new[new_col] = df[col_gest] / (df['dias_mora_cartera'] + 1)
                    df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
                    created.append(new_col)
        
        # Capital por día de mora
        if 'capitalactualsol' in df.columns and 'dias_mora_cartera' in df.columns:
            new_col = 'Capital_por_DiaMora'
            df_new[new_col] = df['capitalactualsol'] / (df['dias_mora_cartera'] + 1)
            df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
            created.append(new_col)
        
        # Gestiones por mes en la empresa
        if 'meses_empresa' in df.columns:
            for period in ['3UM', '6UM']:
                col_gest = f'AVG_GestTotales_CallOut_{period}'
                if col_gest in df.columns:
                    new_col = f'Densidad_Gestiones_por_MesEmpresa_{period}'
                    df_new[new_col] = df[col_gest] / (df['meses_empresa'] + 1)
                    df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
                    created.append(new_col)
        
        self.created_features.extend(created)
        
        if self.verbose:
            print(f"   ✓ Creadas {len(created)} features de densidad")
        
        return df_new
    
    def create_interaction_features(self, df):
        """
        Crea features de interacción entre variables clave.
        """
        if self.verbose:
            print("\n🔧 Creando features de INTERACCIONES...")
        
        df_new = df.copy()
        created = []
        
        # Edad × Días de mora
        if 'Edad' in df.columns and 'dias_mora_cartera' in df.columns:
            new_col = 'Interaccion_Edad_DiaMora'
            df_new[new_col] = df['Edad'] * np.log1p(df['dias_mora_cartera'])
            df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
            created.append(new_col)
        
        # Capital × Días de mora
        if 'capitalactualsol' in df.columns and 'dias_mora_cartera' in df.columns:
            new_col = 'Interaccion_Capital_DiaMora'
            df_new[new_col] = np.log1p(df['capitalactualsol']) * np.log1p(df['dias_mora_cartera'])
            df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
            created.append(new_col)
        
        # Edad × Capital
        if 'Edad' in df.columns and 'capitalactualsol' in df.columns:
            new_col = 'Interaccion_Edad_Capital'
            df_new[new_col] = df['Edad'] * np.log1p(df['capitalactualsol'])
            df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], np.nan)
            created.append(new_col)
        
        self.created_features.extend(created)
        
        if self.verbose:
            print(f"   ✓ Creadas {len(created)} features de interacciones")
        
        return df_new
    
    def create_aggregation_features(self, df):
        """
        Crea agregaciones útiles (sumas, promedios de múltiples periodos).
        """
        if self.verbose:
            print("\n🔧 Creando features de AGREGACIÓN...")
        
        df_new = df.copy()
        created = []
        
        # Total de gestiones efectivas en todos los periodos
        cols_efect = [f'AVG_GestEfect_CallOut_{p}' for p in ['3UM', '6UM', '9UM', '12UM']]
        cols_efect_exist = [c for c in cols_efect if c in df.columns]
        
        if len(cols_efect_exist) > 0:
            new_col = 'Total_GestEfect_AllPeriods'
            df_new[new_col] = df[cols_efect_exist].sum(axis=1)
            created.append(new_col)
        
        # Total de gestiones directas en todos los periodos
        cols_directo = [f'AVG_GestDirecto_CallOut_{p}' for p in ['3UM', '6UM', '9UM', '12UM']]
        cols_directo_exist = [c for c in cols_directo if c in df.columns]
        
        if len(cols_directo_exist) > 0:
            new_col = 'Total_GestDirecto_AllPeriods'
            df_new[new_col] = df[cols_directo_exist].sum(axis=1)
            created.append(new_col)
        
        # Promedio de WhatsApp en todos los periodos
        cols_wpp = [f'AVG_GEST_WPP_OUT_{p}' for p in ['3UM', '6UM', '9UM', '12UM']]
        cols_wpp_exist = [c for c in cols_wpp if c in df.columns]
        
        if len(cols_wpp_exist) > 0:
            new_col = 'Promedio_WhatsApp_AllPeriods'
            df_new[new_col] = df[cols_wpp_exist].mean(axis=1)
            created.append(new_col)
        
        self.created_features.extend(created)
        
        if self.verbose:
            print(f"   ✓ Creadas {len(created)} features de agregación")
        
        return df_new
    
    def create_all_features(self, df, feature_types=['ratio', 'trend', 'density', 'interaction', 'aggregation']):
        """
        Crea todas las features de ingeniería solicitadas.
        
        Args:
            df (DataFrame): Dataset original
            feature_types (list): Tipos de features a crear
            
        Returns:
            DataFrame: Dataset con nuevas features
        """
        if self.verbose:
            print("\n" + "="*100)
            print("🎨 FEATURE ENGINEERING")
            print("="*100)
        
        df_new = df.copy()
        initial_cols = len(df_new.columns)
        
        if 'ratio' in feature_types:
            df_new = self.create_ratio_features(df_new)
        
        if 'trend' in feature_types:
            df_new = self.create_trend_features(df_new)
        
        if 'density' in feature_types:
            df_new = self.create_density_features(df_new)
        
        if 'interaction' in feature_types:
            df_new = self.create_interaction_features(df_new)
        
        if 'aggregation' in feature_types:
            df_new = self.create_aggregation_features(df_new)
        
        final_cols = len(df_new.columns)
        total_created = final_cols - initial_cols
        
        # Manejar NaNs en features creadas
        if self.verbose:
            print(f"\n🔍 Verificando NaNs en features creadas...")
        
        nan_counts = {}
        for feat in self.created_features:
            if feat in df_new.columns:
                nan_count = df_new[feat].isna().sum()
                if nan_count > 0:
                    nan_counts[feat] = nan_count
                    # Rellenar NaNs con la mediana
                    df_new[feat].fillna(df_new[feat].median(), inplace=True)
        
        if nan_counts and self.verbose:
            print(f"   ⚠️  Encontrados NaNs en {len(nan_counts)} features (rellenados con mediana):")
            for feat, count in list(nan_counts.items())[:5]:  # Mostrar solo primeros 5
                print(f"      - {feat}: {count:,} NaNs")
            if len(nan_counts) > 5:
                print(f"      ... y {len(nan_counts) - 5} más")
        elif self.verbose:
            print(f"   ✓ No se encontraron NaNs en features creadas")
        
        if self.verbose:
            print(f"\n✅ Feature Engineering completado:")
            print(f"   • Features originales: {initial_cols}")
            print(f"   • Features creadas: {total_created}")
            print(f"   • Features finales: {final_cols}")
        
        return df_new


class DataPreprocessor:
    """
    Clase para preprocesar datos de cobranza.
    Maneja carga, limpieza, encoding, feature engineering, eliminación de correlaciones y scaling.
    """

    def __init__(self, file_path, random_state=None, apply_feature_engineering=True):
        """
        Args:
            file_path (str): Ruta al archivo de datos
            random_state (int, optional): Semilla aleatoria para reproducibilidad
            apply_feature_engineering (bool): Si aplicar feature engineering
        """
        self.file_path = file_path
        self.random_state = random_state if random_state is not None else np.random.randint(0, 10000)
        self.apply_feature_engineering = apply_feature_engineering
        self.df_original = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.columnas_eliminadas_corr = []
        self.feature_engineer = None
        self.feature_correlations = None  # Para guardar correlaciones con target

        print(f"🎲 Random state: {self.random_state}")
        print(f"🎨 Feature Engineering: {'✓ Activado' if apply_feature_engineering else '✗ Desactivado'}")

    def load_data(self):
        """Carga datos desde archivo txt con delimitador pipe."""
        print(f"📁 Cargando datos desde {self.file_path}...")
        self.df_original = pd.read_csv(self.file_path, sep='|', encoding='utf-8')
        print(f"✓ Datos cargados: {self.df_original.shape}")
        return self

    def preprocess(self):
        """
        Preprocesa los datos: 
        1. Elimina columnas innecesarias
        2. Maneja valores nulos
        3. One-hot encoding
        4. Feature Engineering (ANTES de eliminar correlaciones)
        5. Elimina variables altamente correlacionadas (>0.95)
        """
        print("\n🔧 Preprocesando datos...")

        # Crear copia
        self.df_processed = self.df_original.copy()

        # 1. Eliminar columnas innecesarias
        cols_eliminar = ['CODIGO', 'dfechNac', 'GRUPO_CAPITAL', 'PERIODO']
        self.df_processed = self.df_processed.drop(columns=cols_eliminar, errors='ignore')
        print(f"  ✓ Eliminadas columnas: {[c for c in cols_eliminar if c in self.df_original.columns]}")

        # 2. Manejar valores nulos
        if 'GEnero' in self.df_processed.columns:
            self.df_processed['GEnero'].fillna('Desconocido', inplace=True)

        if 'Estado_civil' in self.df_processed.columns:
            self.df_processed['Estado_civil'].fillna('Desconocido', inplace=True)

        if 'Edad' in self.df_processed.columns:
            median_edad = self.df_processed['Edad'].median()
            self.df_processed['Edad'].fillna(median_edad, inplace=True)

        if 'dias_mora_cartera' in self.df_processed.columns:
            median_mora = self.df_processed['dias_mora_cartera'].median()
            self.df_processed['dias_mora_cartera'].fillna(median_mora, inplace=True)

        print(f"  ✓ Valores nulos manejados")

        # 3. One-Hot Encoding
        cols_categoricas = ['GEnero', 'Estado_civil', 'FLAG_DEPENDIENTE', 'producto_cat', 'GRUPO_CANAL']
        cols_existentes = [col for col in cols_categoricas if col in self.df_processed.columns]

        if cols_existentes:
            self.df_processed = pd.get_dummies(self.df_processed, columns=cols_existentes, drop_first=True)
            print(f"  ✓ One-hot encoding aplicado a: {cols_existentes}")

        # 4. FEATURE ENGINEERING (antes de eliminar correlaciones)
        if self.apply_feature_engineering:
            self.feature_engineer = FeatureEngineer(verbose=True)
            self.df_processed = self.feature_engineer.create_all_features(self.df_processed)
        
        # 5. Eliminar variables altamente correlacionadas (solo entre predictores, no con CD)
        print("\n  🔍 Analizando correlaciones entre variables...")
        
        # Separar target y features
        if 'CD' in self.df_processed.columns:
            X_temp = self.df_processed.drop(columns=['CD'])
            y_temp = self.df_processed['CD']
        else:
            X_temp = self.df_processed.copy()
            y_temp = None
        
        # Calcular matriz de correlación solo de features numéricas
        corr_matrix = X_temp.corr().abs()
        
        # Seleccionar el triángulo superior de la matriz
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Encontrar columnas con correlación > 0.95
        to_drop = [column for column in upper_triangle.columns 
                   if any(upper_triangle[column] > 0.95)]
        
        if to_drop:
            self.columnas_eliminadas_corr = to_drop
            X_temp = X_temp.drop(columns=to_drop)
            print(f"  ✓ Eliminadas {len(to_drop)} variables con correlación > 0.95:")
            for col in to_drop[:10]:  # Mostrar solo primeras 10
                print(f"    - {col}")
            if len(to_drop) > 10:
                print(f"    ... y {len(to_drop) - 10} más")
        else:
            print(f"  ✓ No se encontraron variables con correlación > 0.95")
        
        # Reconstruir dataframe
        if y_temp is not None:
            self.df_processed = pd.concat([X_temp, y_temp], axis=1)
        else:
            self.df_processed = X_temp.copy()

        print(f"\n✓ Preprocesamiento completado: {self.df_processed.shape}")
        print(f"  (Variables reducidas por correlación: {len(self.columnas_eliminadas_corr)})")
        
        return self

    def split_data(self, test_size=0.2, stratify=True):
        """
        Divide datos en train y test, y aplica StandardScaler.

        Args:
            test_size (float): Proporción de datos para test
            stratify (bool): Si mantener proporción de clases
        """
        print(f"\n✂️  Dividiendo datos (test_size={test_size}, stratify={stratify})...")

        X = self.df_processed.drop(columns=['CD'])
        y = self.df_processed['CD']

        stratify_param = y if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        print(f"  Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print(f"  Distribución Train: {dict(self.y_train.value_counts())}")
        print(f"  Distribución Test: {dict(self.y_test.value_counts())}")

        # Aplicar StandardScaler (fit en train, transform en ambos)
        print(f"\n  📏 Aplicando StandardScaler...")
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        print(f"  ✓ StandardScaler aplicado (fit en train, transform en train/test)")

        return self
    
    def analyze_feature_correlations(self, top_n=20, plot=True):
        """
        Analiza y grafica las features que más se correlacionan con el target (CD).
        
        Args:
            top_n (int): Número de top features a mostrar
            plot (bool): Si crear gráfico visual
            
        Returns:
            DataFrame: Top features con sus correlaciones
        """
        print(f"\n📊 Analizando correlaciones de features con el target...")
        
        # Calcular correlaciones con el target
        if 'CD' in self.df_processed.columns:
            correlations = self.df_processed.corr()['CD'].drop('CD').abs().sort_values(ascending=False)
        else:
            print("  ⚠️  No se encontró columna 'CD' (target)")
            return None
        
        # Guardar en el objeto
        self.feature_correlations = correlations
        
        # Obtener top N
        top_features = correlations.head(top_n)
        
        # Identificar features creadas
        created_features = []
        if self.feature_engineer:
            created_features = self.feature_engineer.created_features
        
        # Crear DataFrame con detalles
        df_correlations = pd.DataFrame({
            'Feature': top_features.index,
            'Correlacion_Abs': top_features.values,
            'Es_Creada': [feat in created_features for feat in top_features.index]
        })
        
        # Mostrar tabla
        print(f"\n🏆 TOP {top_n} Features más correlacionadas con CD:")
        print("="*80)
        for idx, row in df_correlations.iterrows():
            marker = "🎨 [CREADA]" if row['Es_Creada'] else "📊 [ORIGINAL]"
            print(f"{idx+1:2d}. {marker:15s} {row['Feature']:45s} | Corr: {row['Correlacion_Abs']:.4f}")
        print("="*80)
        
        # Resumen de features creadas en el top
        creadas_en_top = df_correlations['Es_Creada'].sum()
        print(f"\n✨ Features creadas en el TOP {top_n}: {creadas_en_top} ({creadas_en_top/top_n*100:.1f}%)")
        
        # Crear gráfico
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Colores: azul para originales, verde para creadas
            colors = ['#2ecc71' if feat in created_features else '#3498db' 
                      for feat in top_features.index]
            
            bars = ax.barh(range(len(top_features)), top_features.values, color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features.index, fontsize=9)
            ax.set_xlabel('Correlación Absoluta con CD', fontsize=11, fontweight='bold')
            ax.set_title(f'TOP {top_n} Features Más Correlacionadas con el Target (CD)', 
                        fontsize=13, fontweight='bold', pad=20)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Leyenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label=f'Features Creadas ({creadas_en_top})'),
                Patch(facecolor='#3498db', label=f'Features Originales ({top_n - creadas_en_top})')
            ]
            ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True)
            
            # Añadir valores en las barras
            for i, (bar, val) in enumerate(zip(bars, top_features.values)):
                ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('top_features_correlation_with_target.png', dpi=150, bbox_inches='tight')
            print(f"\n✓ Gráfico guardado en: top_features_correlation_with_target.png")
            plt.show()
        
        return df_correlations

    def get_data(self):
        """Retorna los datos procesados."""
        return self.X_train, self.X_test, self.y_train, self.y_test


class ModelTrainer:
    """
    Clase para entrenar múltiples modelos con diferentes configuraciones.
    Selecciona automáticamente los 3 mejores: Balanceado, Alto Recall, Conservador (ROC-AUC).
    """

    def __init__(self, X_train, y_train, X_test, y_test, random_state=73):
        """
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            random_state: Semilla aleatoria
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.resultados = []
        self.modelos_entrenados = {}
        self.datasets_balanceados = {}

    def _aplicar_tecnicas_balanceo(self):
        """Aplica todas las técnicas de balanceo y guarda los datasets."""
        print("\n" + "="*80)
        print("⚖️  APLICANDO TÉCNICAS DE BALANCEO")
        print("="*80)

        # Dataset original
        self.datasets_balanceados['Original'] = (self.X_train, self.y_train)

        # 1. SMOTE
        print("\n1. Aplicando SMOTE...")
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_smote, y_smote = smote.fit_resample(self.X_train, self.y_train)
        self.datasets_balanceados['SMOTE'] = (X_smote, y_smote)
        print(f"   SMOTE - Shape: {X_smote.shape}, Distribución: {dict(pd.Series(y_smote).value_counts())}")

        # 2. Random Undersampling
        print("\n2. Aplicando Random Undersampling...")
        rus = RandomUnderSampler(random_state=self.random_state, sampling_strategy=0.5)
        X_rus, y_rus = rus.fit_resample(self.X_train, self.y_train)
        self.datasets_balanceados['Undersampling'] = (X_rus, y_rus)
        print(f"   Undersampling - Shape: {X_rus.shape}, Distribución: {dict(pd.Series(y_rus).value_counts())}")

        # 3. SMOTETomek
        print("\n3. Aplicando SMOTETomek...")
        smote_tomek = SMOTETomek(random_state=self.random_state)
        X_st, y_st = smote_tomek.fit_resample(self.X_train, self.y_train)
        self.datasets_balanceados['SMOTETomek'] = (X_st, y_st)
        print(f"   SMOTETomek - Shape: {X_st.shape}, Distribución: {dict(pd.Series(y_st).value_counts())}")

        # 4. ADASYN
        print("\n4. Aplicando ADASYN...")
        adasyn = ADASYN(random_state=self.random_state, n_neighbors=5)
        X_adasyn, y_adasyn = adasyn.fit_resample(self.X_train, self.y_train)
        self.datasets_balanceados['ADASYN'] = (X_adasyn, y_adasyn)
        print(f"   ADASYN - Shape: {X_adasyn.shape}, Distribución: {dict(pd.Series(y_adasyn).value_counts())}")

        print("\n✓ Balanceo completado")

    def train_all_models(self):
        """Entrena múltiples configuraciones de modelos con todos los datasets."""
        print("\n" + "="*80)
        print("🚀 ENTRENANDO ZOO COMPLETO DE MODELOS")
        print("="*80)

        # Aplicar técnicas de balanceo
        self._aplicar_tecnicas_balanceo()

        # Calcular scale_pos_weight
        scale_pos_weight = self.y_train.value_counts()[0] / self.y_train.value_counts()[1]

        # ═══════════════════════════════════════════════════════════════════
        # ZOO COMPLETO DE MODELOS ORGANIZADOS POR CATEGORÍA
        # ═══════════════════════════════════════════════════════════════════
        
        modelos_base = {}
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. MODELOS LINEALES (6 modelos)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['Linear_LogisticRegression'] = (
            'Linear',
            lambda: LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                n_jobs=-1
            )
        )
        
        modelos_base['Linear_LogisticRegression_L1'] = (
            'Linear',
            lambda: LogisticRegression(
                penalty='l1',
                solver='saga',
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                n_jobs=-1
            )
        )
        
        modelos_base['Linear_RidgeClassifier'] = (
            'Linear',
            lambda: RidgeClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
        )
        
        modelos_base['Linear_SGDClassifier'] = (
            'Linear',
            lambda: SGDClassifier(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                n_jobs=-1
            )
        )
        
        modelos_base['Linear_LinearDiscriminant'] = (
            'Linear',
            lambda: LinearDiscriminantAnalysis()
        )
        
        modelos_base['Linear_QuadraticDiscriminant'] = (
            'Linear',
            lambda: QuadraticDiscriminantAnalysis()
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. SVM (2 modelos) - Optimizados para velocidad
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['SVM_LinearSVC'] = (
            'SVM',
            lambda: LinearSVC(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                dual=False  # Más rápido cuando n_samples > n_features
            )
        )
        
        modelos_base['SVM_SVC_Linear'] = (
            'SVM',
            lambda: SVC(
                kernel='linear',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. NAIVE BAYES (2 modelos)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['NaiveBayes_GaussianNB'] = (
            'NaiveBayes',
            lambda: GaussianNB()
        )
        
        modelos_base['NaiveBayes_BernoulliNB'] = (
            'NaiveBayes',
            lambda: BernoulliNB()
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. KNN (2 modelos)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['KNN_K5'] = (
            'KNN',
            lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        )
        
        modelos_base['KNN_K10'] = (
            'KNN',
            lambda: KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. ÁRBOLES DE DECISIÓN (2 modelos)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['Tree_DecisionTree'] = (
            'Trees',
            lambda: DecisionTreeClassifier(
                max_depth=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
        )
        
        modelos_base['Tree_DecisionTree_Deep'] = (
            'Trees',
            lambda: DecisionTreeClassifier(
                max_depth=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 6. ENSEMBLE CLÁSICOS (5 modelos) - Optimizados para velocidad
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['Ensemble_RandomForest'] = (
            'Ensemble_Classic',
            lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        )
        
        modelos_base['Ensemble_ExtraTrees'] = (
            'Ensemble_Classic',
            lambda: ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        )
        
        modelos_base['Ensemble_AdaBoost'] = (
            'Ensemble_Classic',
            lambda: AdaBoostClassifier(
                n_estimators=200,
                random_state=self.random_state,
                algorithm='SAMME'
            )
        )
        
        modelos_base['Ensemble_GradientBoosting'] = (
            'Ensemble_Classic',
            lambda: GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )
        )
        
        modelos_base['Ensemble_Bagging'] = (
            'Ensemble_Classic',
            lambda: BaggingClassifier(
                n_estimators=200,
                max_samples=0.7,
                random_state=self.random_state,
                n_jobs=-1
            )
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 7. GRADIENT BOOSTING AVANZADOS (3 modelos) - Optimizados
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['GradBoost_XGBoost'] = (
            'GradientBoosting',
            lambda: xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='aucpr',
                n_jobs=-1,
                tree_method='hist'
            )
        )
        
        modelos_base['GradBoost_LightGBM'] = (
            'GradientBoosting',
            lambda: lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        )
        
        modelos_base['GradBoost_CatBoost'] = (
            'GradientBoosting',
            lambda: CatBoostClassifier(
                iterations=200,
                depth=7,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=0
            )
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 8. REDES NEURONALES (2 modelos) - Reducidas para velocidad
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        modelos_base['NeuralNet_MLP_Small'] = (
            'NeuralNetwork',
            lambda: MLPClassifier(
                hidden_layer_sizes=(50,),
                max_iter=200,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        )
        
        modelos_base['NeuralNet_MLP_Medium'] = (
            'NeuralNetwork',
            lambda: MLPClassifier(
                hidden_layer_sizes=(50, 25),
                max_iter=200,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 9. MODELOS ESPECÍFICOS PARA DESBALANCE (4 modelos) - imblearn
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if IMBLEARN_ENSEMBLE_AVAILABLE:
            modelos_base['Imbalance_BalancedRandomForest'] = (
                'Imbalance',
                lambda: BalancedRandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # ENTRENAR TODAS LAS COMBINACIONES: MODELOS × DATASETS BALANCEADOS
        # ═══════════════════════════════════════════════════════════════════
        
        total_modelos = len(modelos_base)
        total_datasets = len(self.datasets_balanceados)
        total_combinaciones = total_modelos * total_datasets
        
        print(f"\n📊 Zoo de modelos:")
        print(f"   • Total modelos: {total_modelos}")
        print(f"   • Técnicas de balanceo: {total_datasets}")
        print(f"   • Total combinaciones: {total_combinaciones}")
        print(f"\n  Entrenando todas las combinaciones...")

        for nombre_dataset, (X_train_bal, y_train_bal) in self.datasets_balanceados.items():
            print(f"\n  📊 Dataset: {nombre_dataset}")

            for nombre_modelo, (categoria, modelo_func) in modelos_base.items():
                nombre_completo = f"{nombre_modelo}_{nombre_dataset}"
                print(f"    • {nombre_completo}...", end=' ')

                try:
                    # Crear y entrenar modelo
                    modelo = modelo_func()
                    modelo.fit(X_train_bal, y_train_bal)

                    # Predecir en test
                    y_pred = modelo.predict(self.X_test)
                    
                    # Obtener probabilidades
                    if hasattr(modelo, 'predict_proba'):
                        y_proba = modelo.predict_proba(self.X_test)[:, 1]
                    elif hasattr(modelo, 'decision_function'):
                        y_proba = modelo.decision_function(self.X_test)
                    else:
                        y_proba = y_pred  # Fallback

                    # Calcular métricas
                    resultado = {
                        'nombre': nombre_completo,
                        'modelo_base': nombre_modelo,
                        'categoria': categoria,
                        'tipo_dataset': nombre_dataset,
                        'modelo': modelo,
                        'precision': precision_score(self.y_test, y_pred),
                        'recall': recall_score(self.y_test, y_pred),
                        'f1': f1_score(self.y_test, y_pred),
                        'roc_auc': roc_auc_score(self.y_test, y_proba),
                        'pr_auc': average_precision_score(self.y_test, y_proba),
                        'y_proba': y_proba
                    }

                    self.resultados.append(resultado)
                    self.modelos_entrenados[nombre_completo] = modelo

                    print(f"✓ (R: {resultado['recall']:.3f}, P: {resultado['precision']:.3f}, F1: {resultado['f1']:.3f})")
                
                except Exception as e:
                    print(f"✗ ERROR: {str(e)[:50]}")
                    continue

        print(f"\n✓ {len(self.resultados)} modelos entrenados exitosamente")
        return self

    def select_best_models(self):
        """
        Selecciona los 3 mejores modelos según diferentes criterios:
        1. Balanceado: Mejor F1-Score (balance Precision/Recall)
        2. Alto Recall: Mayor Recall con precision mínima viable (detecta más contactos)
        3. Conservador: Mejor ROC-AUC (capacidad de discriminación general)

        Los 3 modelos serán DIFERENTES (diferente modelo base o técnica de balanceo)
        """
        print("\n" + "="*80)
        print("🎯 SELECCIONANDO MEJORES MODELOS")
        print("="*80)

        df_resultados = pd.DataFrame(self.resultados)

        # 1. Modelo Balanceado (Max F1)
        idx_balanceado = df_resultados['f1'].idxmax()
        modelo_balanceado = df_resultados.iloc[idx_balanceado]
        print(f"\n1️⃣  Modelo Balanceado (Max F1-Score):")
        print(f"    {modelo_balanceado['nombre']}")
        print(f"    F1: {modelo_balanceado['f1']:.4f}, Precision: {modelo_balanceado['precision']:.4f}, Recall: {modelo_balanceado['recall']:.4f}")

        # 2. Modelo Alto Recall (Max Recall, pero con precision >= 10%)
        # Excluir el ya seleccionado
        df_sin_balanceado = df_resultados[df_resultados.index != idx_balanceado]
        df_filtrado_recall = df_sin_balanceado[df_sin_balanceado['precision'] >= 0.10]

        if len(df_filtrado_recall) > 0:
            idx_alto_recall = df_filtrado_recall['recall'].idxmax()
            modelo_alto_recall = df_filtrado_recall.loc[idx_alto_recall]
        else:
            idx_alto_recall = df_sin_balanceado['recall'].idxmax()
            modelo_alto_recall = df_sin_balanceado.loc[idx_alto_recall]

        print(f"\n2️⃣  Modelo Alto Recall (Max Recall con Precision >= 10%):")
        print(f"    {modelo_alto_recall['nombre']}")
        print(f"    Recall: {modelo_alto_recall['recall']:.4f}, Precision: {modelo_alto_recall['precision']:.4f}, F1: {modelo_alto_recall['f1']:.4f}")

        # 3. Modelo Conservador (Max ROC-AUC) - Excluir los 2 anteriores
        df_sin_anteriores = df_resultados[
            (~df_resultados.index.isin([idx_balanceado, idx_alto_recall]))
        ]

        # Filtrar por recall mínimo 50% para que sea útil
        df_filtrado_conservador = df_sin_anteriores[df_sin_anteriores['recall'] >= 0.50]

        if len(df_filtrado_conservador) > 0:
            idx_conservador = df_filtrado_conservador['roc_auc'].idxmax()
            modelo_conservador = df_filtrado_conservador.loc[idx_conservador]
        else:
            # Si no hay con recall >= 50%, buscar mejor ROC-AUC sin filtro
            idx_conservador = df_sin_anteriores['roc_auc'].idxmax()
            modelo_conservador = df_sin_anteriores.loc[idx_conservador]

        print(f"\n3️⃣  Modelo Conservador (Max ROC-AUC con Recall >= 50%):")
        print(f"    {modelo_conservador['nombre']}")
        print(f"    ROC-AUC: {modelo_conservador['roc_auc']:.4f}, Precision: {modelo_conservador['precision']:.4f}, Recall: {modelo_conservador['recall']:.4f}")

        # Guardar selección
        self.mejores_modelos = {
            'Balanceado': {
                'info': modelo_balanceado.to_dict(),
                'modelo': self.modelos_entrenados[modelo_balanceado['nombre']]
            },
            'Alto_Recall': {
                'info': modelo_alto_recall.to_dict(),
                'modelo': self.modelos_entrenados[modelo_alto_recall['nombre']]
            },
            'Conservador': {
                'info': modelo_conservador.to_dict(),
                'modelo': self.modelos_entrenados[modelo_conservador['nombre']]
            }
        }

        # Verificar que son diferentes
        print("\n✓ Verificación de diversidad:")
        nombres = [modelo_balanceado['nombre'], modelo_alto_recall['nombre'], modelo_conservador['nombre']]
        if len(set(nombres)) == 3:
            print("  ✓ Los 3 modelos son diferentes")
        else:
            print("  ⚠️  Algunos modelos coinciden (puede pasar en datasets pequeños)")

        print("\n" + "="*80)
        return self.mejores_modelos

    def get_all_results(self):
        """Retorna todos los resultados de entrenamiento."""
        return pd.DataFrame(self.resultados)


def recall_with_min_precision(y_true, y_pred, min_precision=0.15):
    """
    Scorer personalizado que maximiza recall pero penaliza si precision < min_precision.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        min_precision: Precisión mínima aceptable (default 0.15 = 15%)
    
    Returns:
        float: Score combinado
    """
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    # Si la precisión es menor al mínimo, penalizar fuertemente
    if prec < min_precision:
        # Penalización proporcional a qué tan lejos está del mínimo
        penalty = (min_precision - prec) * 2  # Factor 2 para penalizar más
        return rec - penalty
    else:
        # Si cumple el mínimo, retornar recall puro
        return rec


class ModelTuner:
    """
    Clase para hacer Hyperparameter Tuning de los mejores modelos seleccionados.
    Usa RandomizedSearchCV para buscar los mejores hiperparámetros.
    
    Estrategias de tuning:
    - Balanceado: Optimiza F1-score (equilibrio precision/recall)
    - Alto_Recall: Optimiza Recall con restricción de precision >= 15%
    - Conservador: Optimiza ROC-AUC (mejor discriminación)
    """

    def __init__(self, X_train, y_train, X_test, y_test, random_state=73):
        """
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de test
            y_test: Target de test
            random_state: Semilla aleatoria
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.tuned_models = {}
        
    def _get_param_distributions(self, modelo_base):
        """
        Retorna distribuciones de parámetros para cada tipo de modelo.
        """
        # Determinar tipo de modelo
        model_type = type(modelo_base).__name__
        
        if 'RandomForest' in model_type:
            return {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [5, 7, 10, 15, 20,  None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        
        elif 'XGB' in model_type or 'XGBoost' in model_type:
            return {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [3, 5, 7, 10, 15, 20],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0, 0.1, 0.2, 0.3],
                'scale_pos_weight': [1, 2, 3, 5, 10]
            }
        
        elif 'LightGBM' in model_type or 'LGBM' in model_type:
            return {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [3, 5, 7, 10, 15, 20, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                'num_leaves': [15, 31, 63, 127, 255],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [5, 10, 20, 30, 50],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0],
                'scale_pos_weight': [1, 2, 3, 5, 10]
            }
        
        elif 'CatBoost' in model_type:
            return {
                'iterations': [50, 100, 150, 200, 250, 300],
                'depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'border_count': [32, 64, 128, 255],
                'scale_pos_weight': [1, 2, 3, 5, 10]
            }
        
        elif 'GradientBoosting' in model_type:
            return {
                'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [3, 5, 7, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None]
            }
        
        elif 'ExtraTrees' in model_type:
            return {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [5, 10, 15, 20, 25, 30, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None]
            }
        
        else:
            # Para otros modelos, retornar None (no tunear)
            return None
    
    def tune_model(self, modelo_base, nombre, n_iter=150, cv=3, scoring='f1', min_precision=None):
        """
        Hace tuning de un modelo usando RandomizedSearchCV.
        
        Args:
            modelo_base: Modelo base a tunear
            nombre: Nombre del modelo
            n_iter: Número de iteraciones de búsqueda aleatoria
            cv: Número de folds para cross-validation
            scoring: Métrica a optimizar ('f1', 'recall', 'roc_auc')
            min_precision: Si se optimiza recall, precisión mínima aceptable (ej: 0.15 para 15%)
            
        Returns:
            dict: Información del mejor modelo encontrado
        """
        print(f"\n{'='*80}")
        print(f"🔧 TUNING: {nombre}")
        print(f"{'='*80}")
        
        # Obtener distribución de parámetros
        param_dist = self._get_param_distributions(modelo_base)
        
        if param_dist is None:
            print(f"⚠️  No hay configuración de tuning para {type(modelo_base).__name__}")
            print(f"   Usando modelo original sin tuning")
            
            # Entrenar modelo original y evaluar
            modelo_base.fit(self.X_train, self.y_train)
            y_pred = modelo_base.predict(self.X_test)
            y_proba = modelo_base.predict_proba(self.X_test)[:, 1] if hasattr(modelo_base, 'predict_proba') else None
            
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_proba) if y_proba is not None else 0.0
            
            return {
                'modelo': modelo_base,
                'mejor_score': f1,
                'mejores_params': {},
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'tuneado': False
            }
        
        print(f"📊 Configuración:")
        print(f"   • Iteraciones: {n_iter}")
        print(f"   • Cross-validation: {cv} folds")
        print(f"   • Métrica: {scoring}")
        if min_precision is not None:
            print(f"   • Restricción: Precision >= {min_precision*100:.0f}%")
        print(f"   • Parámetros a explorar: {len(param_dist)}")
        print(f"\n⏳ Buscando mejores hiperparámetros... (puede tomar varios minutos)")
        
        # Si se requiere recall con precision mínima, crear scorer personalizado
        if scoring == 'recall' and min_precision is not None:
            from functools import partial
            custom_scorer = make_scorer(
                partial(recall_with_min_precision, min_precision=min_precision),
                greater_is_better=True
            )
            scoring_to_use = custom_scorer
            print(f"   🎯 Usando scorer personalizado: Recall con Precision >= {min_precision*100:.0f}%")
        else:
            scoring_to_use = scoring
        
        # Crear RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=modelo_base,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring_to_use,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # Entrenar
        random_search.fit(self.X_train, self.y_train)
        
        # Mejor modelo
        best_model = random_search.best_estimator_
        
        # Evaluar en test
        y_pred = best_model.predict(self.X_test)
        
        # Obtener probabilidades
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(self.X_test)[:, 1]
        elif hasattr(best_model, 'decision_function'):
            y_proba = best_model.decision_function(self.X_test)
        else:
            y_proba = y_pred.astype(float)
        
        # Métricas
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        print(f"\n✅ TUNING COMPLETADO")
        print(f"   • Mejor score CV: {random_search.best_score_:.4f}")
        print(f"   • Test - Precision: {precision:.4f}")
        print(f"   • Test - Recall: {recall:.4f}")
        print(f"   • Test - F1: {f1:.4f}")
        print(f"   • Test - ROC-AUC: {roc_auc:.4f}")
        
        print(f"\n📋 Mejores parámetros encontrados:")
        for param, value in random_search.best_params_.items():
            print(f"   • {param}: {value}")
        
        print(f"{'='*80}")
        
        return {
            'modelo': best_model,
            'mejor_score': random_search.best_score_,
            'mejores_params': random_search.best_params_,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_proba': y_proba,
            'tuneado': True
        }
    
    def tune_best_models(self, mejores_modelos, n_iter=150):
        """
        Tunea los 3 mejores modelos seleccionados, usando métricas específicas para cada enfoque:
        - Balanceado: F1-score (equilibrio precision/recall)
        - Alto_Recall: Recall (maximizar detección, con precision mínima 15%)
        - Conservador: ROC-AUC (mejor discriminación)
        
        Args:
            mejores_modelos: Diccionario con los 3 mejores modelos
            n_iter: Número de iteraciones por modelo
            
        Returns:
            dict: Modelos tuneados con sus métricas mejoradas
        """
        print("\n" + "="*80)
        print("🔧 HYPERPARAMETER TUNING DE LOS 3 MEJORES MODELOS")
        print("="*80)
        print(f"\n📊 Configuración:")
        print(f"   • Modelos a tunear: {len(mejores_modelos)}")
        print(f"   • Iteraciones por modelo: {n_iter}")
        print(f"   • Total de combinaciones a probar: {len(mejores_modelos) * n_iter}")
        print(f"   • Estrategia de tuning:")
        print(f"     - Balanceado: Optimiza F1-score (balance precision/recall)")
        print(f"     - Alto_Recall: Optimiza Recall (con precision >= 15%)")
        print(f"     - Conservador: Optimiza ROC-AUC (mejor discriminación)")
        print(f"   • Tiempo estimado: 10-30 minutos (depende del hardware)")
        
        resultados_tuning = {}
        
        for i, (tipo, data) in enumerate(mejores_modelos.items(), 1):
            modelo_original = data['modelo']
            info_original = data['info']
            
            print(f"\n{'─'*80}")
            print(f"📌 [{i}/3] Tuneando: {tipo}")
            print(f"   Modelo original: {info_original['nombre']}")
            print(f"   Métricas originales - F1: {info_original['f1']:.4f}, Recall: {info_original['recall']:.4f}, Precision: {info_original['precision']:.4f}")
            
            # Determinar métrica de tuning según el tipo de modelo
            if tipo == 'Balanceado':
                scoring = 'f1'
                min_prec = None
                print(f"   🎯 Estrategia: Maximizar F1-score")
            elif tipo == 'Alto_Recall':
                scoring = 'recall'
                min_prec = 0.15  # Precision mínima del 15%
                print(f"   🎯 Estrategia: Maximizar Recall (con precision >= 15%)")
            elif tipo == 'Conservador':
                scoring = 'roc_auc'
                min_prec = None
                print(f"   🎯 Estrategia: Maximizar ROC-AUC")
            else:
                scoring = 'f1'
                min_prec = None
                print(f"   🎯 Estrategia: Usar F1-score por defecto")
            
            print(f"{'─'*80}")
            
            # Tunear modelo con la métrica apropiada
            resultado = self.tune_model(
                modelo_base=modelo_original,
                nombre=f"{tipo} - {info_original['nombre']}",
                n_iter=n_iter,
                cv=2,
                scoring=scoring,
                min_precision=min_prec
            )
            
            # Comparar mejora
            mejora_f1 = resultado['f1'] - info_original['f1']
            mejora_recall = resultado['recall'] - info_original['recall']
            mejora_precision = resultado['precision'] - info_original['precision']
            
            print(f"\n📊 COMPARACIÓN:")
            print(f"   • F1:        {info_original['f1']:.4f} → {resultado['f1']:.4f} ({mejora_f1:+.4f})")
            print(f"   • Recall:    {info_original['recall']:.4f} → {resultado['recall']:.4f} ({mejora_recall:+.4f})")
            print(f"   • Precision: {info_original['precision']:.4f} → {resultado['precision']:.4f} ({mejora_precision:+.4f})")
            
            if mejora_f1 > 0:
                print(f"   ✅ MEJORA del {(mejora_f1/info_original['f1']*100):+.2f}% en F1")
            else:
                print(f"   ⚠️  Sin mejora significativa (puede usar modelo original)")
            
            # Actualizar info
            info_tuneada = info_original.copy()
            info_tuneada['precision'] = resultado['precision']
            info_tuneada['recall'] = resultado['recall']
            info_tuneada['f1'] = resultado['f1']
            info_tuneada['roc_auc'] = resultado['roc_auc']
            info_tuneada['nombre'] = info_original['nombre'] + ' (TUNED)'
            info_tuneada['mejores_params'] = resultado['mejores_params']
            info_tuneada['tuneado'] = resultado['tuneado']
            
            resultados_tuning[tipo] = {
                'modelo': resultado['modelo'],
                'info': info_tuneada,
                'modelo_original': modelo_original,
                'info_original': info_original,
                'mejora_f1': mejora_f1
            }
        
        print(f"\n{'='*80}")
        print("✅ TUNING COMPLETADO PARA TODOS LOS MODELOS")
        print(f"{'='*80}")
        
        # Resumen final
        print(f"\n📊 RESUMEN DE MEJORAS:")
        for tipo, data in resultados_tuning.items():
            mejora = data['mejora_f1']
            simbolo = "✅" if mejora > 0 else "⚠️"
            print(f"   {simbolo} {tipo}: F1 {mejora:+.4f} ({(mejora/data['info_original']['f1']*100):+.2f}%)")
        
        print(f"\n{'='*80}")
        
        self.tuned_models = resultados_tuning
        return resultados_tuning


class ContactSimulator:
    """
    Clase para simular el proceso real de llamadas en cobranza.
    Ordena intentos por probabilidad y mide cuántas llamadas se necesitan
    para encontrar diferentes porcentajes de contactos.
    """

    def __init__(self, X_test, y_test, random_state=73):
        """
        Args:
            X_test: Features de test
            y_test: Labels de test (contactos reales)
            random_state: Semilla aleatoria
        """
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.total_contactos = y_test.sum()
        self.total_intentos = len(y_test)

        print(f"\n📊 Configuración de simulación:")
        print(f"  Total intentos: {self.total_intentos:,}")
        print(f"  Total contactos reales: {self.total_contactos:,} ({self.total_contactos/self.total_intentos*100:.2f}%)")

    def _calcular_score_empresarial(self):
        """
        Calcula score de priorización según criterio empresarial:
        1. Mayor capital (más importante)
        2. Mayor edad (más importante) 
        3. Dependiente > Independiente (más importante)
        
        Returns:
            np.array: Score de priorización (mayor = más prioritario)
        """
        # Obtener columnas necesarias del dataset original
        X_array = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
        columnas = self.X_test.columns.tolist() if hasattr(self.X_test, 'columns') else []
        
        # Identificar índices de las columnas clave
        capital_idx = columnas.index('capitalactualsol') if 'capitalactualsol' in columnas else None
        edad_idx = columnas.index('Edad') if 'Edad' in columnas else None
        dependiente_idx = columnas.index('FLAG_DEPENDIENTE') if 'FLAG_DEPENDIENTE' in columnas else None
        
        # Extraer valores (con fallback si no existen)
        if capital_idx is not None:
            capital = X_array[:, capital_idx]
        else:
            capital = np.zeros(len(self.y_test))
        
        if edad_idx is not None:
            edad = X_array[:, edad_idx]
        else:
            edad = np.zeros(len(self.y_test))
        
        if dependiente_idx is not None:
            es_dependiente = X_array[:, dependiente_idx]
        else:
            es_dependiente = np.zeros(len(self.y_test))
        
        # Normalizar valores para que estén en escala comparable
        # Capital: normalizar entre 0-1
        capital_norm = (capital - capital.min()) / (capital.max() - capital.min() + 1e-6)
        
        # Edad: normalizar entre 0-1
        edad_norm = (edad - edad.min()) / (edad.max() - edad.min() + 1e-6)
        
        # Dependiente: ya es 0 o 1
        dependiente_norm = es_dependiente
        
        # Score combinado con pesos según importancia empresarial
        # Capital (50%), Edad (30%), Dependiente (20%)
        score = (capital_norm * 0.50) + (edad_norm * 0.30) + (dependiente_norm * 0.20)
        
        return score
    
    def simular_estrategia(self, nombre, y_proba=None, usar_criterio_empresarial=False):
        """
        Simula una estrategia de llamadas ordenando por probabilidad o criterio empresarial.

        Args:
            nombre (str): Nombre de la estrategia
            y_proba (array, optional): Probabilidades de contacto. Si None, usa orden aleatorio.
            usar_criterio_empresarial (bool): Si True, usa capital+edad+dependiente para ordenar

        Returns:
            dict: Resultados de la simulación
        """
        # Determinar el score a usar para ordenar
        if usar_criterio_empresarial:
            score_orden = self._calcular_score_empresarial()
        elif y_proba is not None:
            score_orden = y_proba
        else:
            score_orden = np.random.random(len(self.y_test))
        
        # Crear DataFrame con índices originales
        df_sim = pd.DataFrame({
            'idx_original': range(len(self.y_test)),
            'contacto_real': self.y_test.values,
            'proba': score_orden
        })

        # Ordenar por score (descendente)
        df_sim = df_sim.sort_values('proba', ascending=False).reset_index(drop=True)

        # Calcular contactos acumulados
        df_sim['contactos_acum'] = df_sim['contacto_real'].cumsum()
        df_sim['porcentaje_contactos'] = (df_sim['contactos_acum'] / self.total_contactos) * 100
        df_sim['llamadas_realizadas'] = range(1, len(df_sim) + 1)
        df_sim['porcentaje_llamadas'] = (df_sim['llamadas_realizadas'] / self.total_intentos) * 100

        # Encontrar puntos de interés: cuántas llamadas para encontrar X% de contactos
        hitos = [50, 80, 95, 99, 100]
        llamadas_en_hito = {}

        for hito in hitos:
            # Encontrar primera fila donde alcanzamos el hito
            rows = df_sim[df_sim['porcentaje_contactos'] >= hito]
            if len(rows) > 0:
                llamadas = rows.iloc[0]['llamadas_realizadas']
                porcentaje_dataset = rows.iloc[0]['porcentaje_llamadas']
                llamadas_en_hito[hito] = {
                    'llamadas': llamadas,
                    'porcentaje_dataset': porcentaje_dataset
                }
            else:
                llamadas_en_hito[hito] = {
                    'llamadas': self.total_intentos,
                    'porcentaje_dataset': 100.0
                }

        # Calcular métricas adicionales
        # Tasa de éxito en primeras N llamadas
        top_10_pct = int(self.total_intentos * 0.10)
        top_20_pct = int(self.total_intentos * 0.20)
        top_50_pct = int(self.total_intentos * 0.50)

        contactos_top10 = df_sim.iloc[:top_10_pct]['contacto_real'].sum()
        contactos_top20 = df_sim.iloc[:top_20_pct]['contacto_real'].sum()
        contactos_top50 = df_sim.iloc[:top_50_pct]['contacto_real'].sum()

        return {
            'nombre': nombre,
            'hitos': llamadas_en_hito,
            'tasa_top_10': contactos_top10 / top_10_pct * 100,
            'tasa_top_20': contactos_top20 / top_20_pct * 100,
            'tasa_top_50': contactos_top50 / top_50_pct * 100,
            'df_simulacion': df_sim,
            'y_proba': df_sim['proba'].values
        }

    def comparar_estrategias(self, modelos_dict):
        """
        Compara múltiples estrategias incluyendo aleatorio y criterio empresarial.

        Args:
            modelos_dict (dict): Diccionario con modelos {'nombre': modelo_entrenado}

        Returns:
            dict: Resultados de todas las simulaciones
        """
        print("\n" + "="*80)
        print("🎮 SIMULANDO ESTRATEGIAS DE LLAMADAS")
        print("="*80)

        resultados = {}

        # 1. Simular estrategia EMPRESARIAL (Capital → Edad → Dependiente)
        print("\n  📌 Simulando: Criterio_Empresarial (Capital→Edad→Dependiente)...", end=' ')
        resultados['Criterio_Empresarial'] = self.simular_estrategia(
            'Criterio_Empresarial',
            y_proba=None,
            usar_criterio_empresarial=True
        )
        print("✓")

        # 2. Simular estrategia aleatoria (baseline)
        print("  📌 Simulando: Aleatorio (Baseline)...", end=' ')
        np.random.seed(self.random_state)
        resultados['Aleatorio'] = self.simular_estrategia(
            'Aleatorio',
            y_proba=None,
            usar_criterio_empresarial=False
        )
        print("✓")

        # 3. Simular cada modelo ML
        for nombre, modelo in modelos_dict.items():
            print(f"  📌 Simulando: {nombre}...", end=' ')
            y_proba = modelo.predict_proba(self.X_test)[:, 1]
            resultados[nombre] = self.simular_estrategia(
                nombre,
                y_proba=y_proba,
                usar_criterio_empresarial=False
            )
            print("✓")

        # Crear tabla comparativa
        print("\n" + "="*80)
        print("📊 RESULTADOS: Llamadas necesarias para encontrar X% de contactos")
        print("="*80)

        # Header
        print(f"\n{'Estrategia':<25} | {'50%':<12} | {'80%':<12} | {'95%':<12} | {'99%':<12} | {'100%':<12}")
        print("-" * 105)

        for nombre, resultado in resultados.items():
            row = f"{nombre:<25}"
            for hito in [50, 80, 95, 99, 100]:
                llamadas = resultado['hitos'][hito]['llamadas']
                pct = resultado['hitos'][hito]['porcentaje_dataset']
                row += f" | {llamadas:>5,} ({pct:>4.1f}%)"
            print(row)

        # Tabla de tasas de éxito
        print("\n" + "="*80)
        print("📊 TASA DE ÉXITO: % de contactos en primeras X% de llamadas")
        print("="*80)
        print(f"\n{'Estrategia':<25} | {'Top 10%':<10} | {'Top 20%':<10} | {'Top 50%':<10}")
        print("-" * 70)

        for nombre, resultado in resultados.items():
            print(f"{nombre:<25} | {resultado['tasa_top_10']:>8.2f}% | "
                  f"{resultado['tasa_top_20']:>8.2f}% | {resultado['tasa_top_50']:>8.2f}%")

        return resultados


class MetricsCalculator:
    """
    Clase para calcular métricas de negocio relevantes.
    """

    @staticmethod
    def calcular_metricas_negocio(resultados_simulacion, total_contactos):
        """
        Calcula métricas de negocio para cada estrategia.

        Args:
            resultados_simulacion (dict): Resultados de simulación
            total_contactos (int): Total de contactos reales

        Returns:
            DataFrame: Métricas de negocio por estrategia
        """
        print("\n" + "="*80)
        print("💼 MÉTRICAS DE NEGOCIO")
        print("="*80)

        metricas = []

        for nombre, resultado in resultados_simulacion.items():
            # Llamadas para encontrar todos los contactos
            llamadas_100 = resultado['hitos'][100]['llamadas']
            pct_dataset_100 = resultado['hitos'][100]['porcentaje_dataset']

            # Llamadas para encontrar 80% (objetivo más realista)
            llamadas_80 = resultado['hitos'][80]['llamadas']
            pct_dataset_80 = resultado['hitos'][80]['porcentaje_dataset']

            # Contactos detectados en primera mitad
            contactos_50 = int(total_contactos * resultado['hitos'][50]['porcentaje_dataset'] / 100)

            metricas.append({
                'Estrategia': nombre,
                'Llamadas_para_80%': llamadas_80,
                'Llamadas_para_100%': llamadas_100,
                'Llamadas_por_contacto_80%': llamadas_80 / (total_contactos * 0.80),
                'Llamadas_por_contacto_100%': llamadas_100 / total_contactos,
                'Tasa_exito_top10%': resultado['tasa_top_10'],
                'Tasa_exito_top20%': resultado['tasa_top_20'],
                'Dataset_usado_80%': pct_dataset_80,
                'Dataset_usado_100%': pct_dataset_100
            })

        df_metricas = pd.DataFrame(metricas)

        # Mostrar
        print("\n" + df_metricas.to_string(index=False))

        # Comparar con aleatorio
        if 'Aleatorio' in resultados_simulacion:
            aleatorio = df_metricas[df_metricas['Estrategia'] == 'Aleatorio'].iloc[0]

            print("\n" + "="*80)
            print("📈 MEJORA vs ALEATORIO (Objetivo 80% de contactos)")
            print("="*80)

            for _, row in df_metricas.iterrows():
                if row['Estrategia'] != 'Aleatorio':
                    reduccion = (aleatorio['Llamadas_para_80%'] - row['Llamadas_para_80%'])
                    pct_mejora = (reduccion / aleatorio['Llamadas_para_80%']) * 100
                    print(f"\n{row['Estrategia']}:")
                    print(f"  • Reduce {reduccion:,.0f} llamadas ({pct_mejora:+.1f}%)")
                    print(f"  • Tasa de éxito Top 10%: {row['Tasa_exito_top10%']:.2f}% (vs {aleatorio['Tasa_exito_top10%']:.2f}%)")

        return df_metricas


def get_best_models_summary(mejores_modelos):
    """
    Crea un resumen de los mejores modelos seleccionados.

    Args:
        mejores_modelos (dict): Diccionario con los 3 mejores modelos

    Returns:
        DataFrame: Resumen de modelos
    """
    data = []
    for tipo, info in mejores_modelos.items():
        data.append({
            'Tipo': tipo,
            'Nombre': info['info']['nombre'],
            'Dataset': info['info']['tipo_dataset'],
            'Precision': info['info']['precision'],
            'Recall': info['info']['recall'],
            'F1-Score': info['info']['f1'],
            'ROC-AUC': info['info']['roc_auc']
        })

    return pd.DataFrame(data)
