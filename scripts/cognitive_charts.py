import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = dataset

# Configuración de estilo
plt.style.use('seaborn-v0_8')
custom_colors = ['#f9afaf', '#414b3b', '#f4f4f4', '#232020']
sns.set_palette(custom_colors)

class CognitiveCharts:
    """
    Clase para generar gráficos de análisis cognitivo compatibles con Power BI
    """
    
    def __init__(self, df):
        """
        Inicializar con el DataFrame de datos cognitivos
        
        Args:
            df (pandas.DataFrame): DataFrame con los datos de human_cognitive
        """
        self.df = df
        self.numeric_cols = ['Age', 'AI_Predicted_Score', 'Caffeine_Intake', 
                            'Cognitive_Score', 'Daily_Screen_Time', 'Memory_Test_Score',
                            'Reaction_Time', 'Sleep_Duration', 'Stress_Level']
        self.categorical_cols = ['Diet_Type', 'Exercise_Frequency', 'Gender']
    
    # =========================
    # GRÁFICOS DE DISTRIBUCIÓN
    # =========================
    
    def plot_cognitive_distribution(self, save_path=None, show_plot=True):
        """
        Histograma de distribución de puntuaciones cognitivas con estadísticas
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma
        ax1.hist(self.df['Cognitive_Score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.df['Cognitive_Score'].mean(), color='red', linestyle='--', 
                label=f'Media: {self.df["Cognitive_Score"].mean():.2f}')
        ax1.axvline(self.df['Cognitive_Score'].median(), color='green', linestyle='--',
                label=f'Mediana: {self.df["Cognitive_Score"].median():.2f}')
        ax1.set_xlabel('Puntuación Cognitiva')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Puntuaciones Cognitivas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(self.df['Cognitive_Score'], vert=True)
        ax2.set_ylabel('Puntuación Cognitiva')
        ax2.set_title('Box Plot - Puntuaciones Cognitivas')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_age_distribution_by_gender(self, save_path=None, show_plot=True):
        """
        Distribución de edad por género
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for gender in self.df['Gender'].unique():
            data = self.df[self.df['Gender'] == gender]['Age']
            ax.hist(data, alpha=0.6, label=f'{gender} (n={len(data)})', bins=20)
        
        ax.set_xlabel('Edad')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Edad por Género')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
            
        return fig
    
    # =========================
    # ANÁLISIS COMPARATIVO
    # =========================
    
    def plot_cognitive_by_categories(self, save_path=None, show_plot=True):
        """
        Puntuaciones cognitivas por categorías demográficas
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Por género
        sns.boxplot(data=self.df, x='Gender', y='Cognitive_Score', ax=axes[0,0])
        axes[0,0].set_title('Puntuación Cognitiva por Género')
        axes[0,0].grid(True, alpha=0.3)
        
        # Por tipo de dieta
        sns.boxplot(data=self.df, x='Diet_Type', y='Cognitive_Score', ax=axes[0,1])
        axes[0,1].set_title('Puntuación Cognitiva por Tipo de Dieta')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Por frecuencia de ejercicio
        sns.boxplot(data=self.df, x='Exercise_Frequency', y='Cognitive_Score', ax=axes[1,0])
        axes[1,0].set_title('Puntuación Cognitiva por Frecuencia de Ejercicio')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Por grupos de edad
        self.df['Age_Group'] = pd.cut(self.df['Age'], bins=[0, 30, 50, 100], 
                                    labels=['18-30', '31-50', '50+'])
        sns.boxplot(data=self.df, x='Age_Group', y='Cognitive_Score', ax=axes[1,1])
        axes[1,1].set_title('Puntuación Cognitiva por Grupo de Edad')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
            
        return fig
    
    def plot_performance_comparison(self, save_path=None, show_plot=True):
        """
        Comparación de diferentes métricas de rendimiento
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cognitive vs Memory Score
        axes[0,0].scatter(self.df['Cognitive_Score'], self.df['Memory_Test_Score'], 
                        alpha=0.6, color='blue')
        z = np.polyfit(self.df['Cognitive_Score'], self.df['Memory_Test_Score'], 1)
        p = np.poly1d(z)
        axes[0,0].plot(self.df['Cognitive_Score'], p(self.df['Cognitive_Score']), 
                    "r--", alpha=0.8)
        axes[0,0].set_xlabel('Puntuación Cognitiva')
        axes[0,0].set_ylabel('Puntuación Test Memoria')
        axes[0,0].set_title('Cognitive Score vs Memory Test Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # Predicted vs Actual
        axes[0,1].scatter(self.df['AI_Predicted_Score'], self.df['Cognitive_Score'], 
                        alpha=0.6, color='green')
        # Línea perfecta de predicción
        min_val = min(self.df['AI_Predicted_Score'].min(), self.df['Cognitive_Score'].min())
        max_val = max(self.df['AI_Predicted_Score'].max(), self.df['Cognitive_Score'].max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0,1].set_xlabel('AI Predicted Score')
        axes[0,1].set_ylabel('Cognitive Score Actual')
        axes[0,1].set_title('Predicción vs Realidad')
        axes[0,1].grid(True, alpha=0.3)
        
        # Age vs Reaction Time
        axes[1,0].scatter(self.df['Age'], self.df['Reaction_Time'], alpha=0.6, color='orange')
        z = np.polyfit(self.df['Age'], self.df['Reaction_Time'], 1)
        p = np.poly1d(z)
        axes[1,0].plot(self.df['Age'], p(self.df['Age']), "r--", alpha=0.8)
        axes[1,0].set_xlabel('Edad')
        axes[1,0].set_ylabel('Tiempo de Reacción')
        axes[1,0].set_title('Edad vs Tiempo de Reacción')
        axes[1,0].grid(True, alpha=0.3)
        
        # Sleep vs Cognitive Score
        axes[1,1].scatter(self.df['Sleep_Duration'], self.df['Cognitive_Score'], 
                        alpha=0.6, color='purple')
        z = np.polyfit(self.df['Sleep_Duration'], self.df['Cognitive_Score'], 1)
        p = np.poly1d(z)
        axes[1,1].plot(self.df['Sleep_Duration'], p(self.df['Sleep_Duration']), 
                    "r--", alpha=0.8)
        axes[1,1].set_xlabel('Duración del Sueño (horas)')
        axes[1,1].set_ylabel('Puntuación Cognitiva')
        axes[1,1].set_title('Sueño vs Rendimiento Cognitivo')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
            
        return fig
    
    # =========================
    # MATRIZ DE CORRELACIÓN
    # =========================
    
    def plot_correlation_matrix(self, save_path=None, show_plot=True):
        """
        Matriz de correlación de variables numéricas
        """
        # Calcular matriz de correlación
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Matriz de Correlación - Variables Cognitivas', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
            
        return fig
    
    # =========================
    # ANÁLISIS DE FACTORES DE RIESGO
    # =========================
    
    def plot_lifestyle_impact(self, save_path=None, show_plot=True):
        """
        Impacto de factores de estilo de vida en el rendimiento cognitivo
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Screen Time vs Cognitive Score
        screen_time_bins = pd.cut(self.df['Daily_Screen_Time'], bins=4, labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
        screen_data = self.df.groupby(screen_time_bins)['Cognitive_Score'].mean()
        axes[0,0].bar(screen_data.index, screen_data.values, color='coral')
        axes[0,0].set_xlabel('Tiempo de Pantalla Diario')
        axes[0,0].set_ylabel('Puntuación Cognitiva Promedio')
        axes[0,0].set_title('Impacto del Tiempo de Pantalla')
        axes[0,0].grid(True, alpha=0.3)
        
        # Stress Level vs Cognitive Score
        stress_bins = pd.cut(self.df['Stress_Level'], bins=3, labels=['Bajo', 'Medio', 'Alto'])
        stress_data = self.df.groupby(stress_bins)['Cognitive_Score'].mean()
        axes[0,1].bar(stress_data.index, stress_data.values, color='lightcoral')
        axes[0,1].set_xlabel('Nivel de Estrés')
        axes[0,1].set_ylabel('Puntuación Cognitiva Promedio')
        axes[0,1].set_title('Impacto del Estrés')
        axes[0,1].grid(True, alpha=0.3)
        
        # Caffeine Intake vs Memory Score
        caffeine_bins = pd.cut(self.df['Caffeine_Intake'], bins=4, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto'])
        caffeine_data = self.df.groupby(caffeine_bins)['Memory_Test_Score'].mean()
        axes[1,0].bar(caffeine_data.index, caffeine_data.values, color='saddlebrown')
        axes[1,0].set_xlabel('Consumo de Cafeína')
        axes[1,0].set_ylabel('Puntuación Test Memoria Promedio')
        axes[1,0].set_title('Impacto de la Cafeína en Memoria')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Sleep Duration vs Reaction Time
        sleep_bins = pd.cut(self.df['Sleep_Duration'], bins=4, labels=['<6h', '6-7h', '7-8h', '>8h'])
        sleep_data = self.df.groupby(sleep_bins)['Reaction_Time'].mean()
        axes[1,1].bar(sleep_data.index, sleep_data.values, color='mediumpurple')
        axes[1,1].set_xlabel('Duración del Sueño')
        axes[1,1].set_ylabel('Tiempo de Reacción Promedio')
        axes[1,1].set_title('Impacto del Sueño en Reacción')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
            
        return fig
    
    # =========================
    # GRÁFICOS INTERACTIVOS (PLOTLY)
    # =========================
    
    def create_interactive_dashboard(self, save_html_path=None):
        """
        Dashboard interactivo con Plotly para Power BI
        """
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución Cognitiva', 'Correlación Edad-Cognición',
                        'Rendimiento por Género', 'Predicción vs Realidad'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histograma de puntuaciones cognitivas
        fig.add_trace(
            go.Histogram(x=self.df['Cognitive_Score'], name='Cognitive Score',
                        marker_color='lightblue', opacity=0.7),
            row=1, col=1
        )
        
        # Scatter Age vs Cognitive Score
        fig.add_trace(
            go.Scatter(x=self.df['Age'], y=self.df['Cognitive_Score'],
                    mode='markers', name='Age vs Cognitive',
                    marker=dict(color='orange', opacity=0.6)),
            row=1, col=2
        )
        
        # Box plot por género
        for gender in self.df['Gender'].unique():
            data = self.df[self.df['Gender'] == gender]['Cognitive_Score']
            fig.add_trace(
                go.Box(y=data, name=f'{gender}', boxpoints='outliers'),
                row=2, col=1
            )
        
        # Predicted vs Actual
        fig.add_trace(
            go.Scatter(x=self.df['AI_Predicted_Score'], y=self.df['Cognitive_Score'],
                    mode='markers', name='Predicted vs Actual',
                    marker=dict(color='green', opacity=0.6)),
            row=2, col=2
        )
        
        # Línea de predicción perfecta
        min_val = min(self.df['AI_Predicted_Score'].min(), self.df['Cognitive_Score'].min())
        max_val = max(self.df['AI_Predicted_Score'].max(), self.df['Cognitive_Score'].max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='Perfect Prediction',
                    line=dict(color='red', dash='dash')),
        row=2, col=2
        )
        
        # Actualizar layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Dashboard Interactivo - Análisis Cognitivo",
            title_x=0.5
        )
        
        if save_html_path:
            fig.write_html(save_html_path)
        
        return fig
    
    # =========================
    # FUNCIÓN PARA GENERAR KPIs
    # =========================
    
    def generate_kpis(self):
        """
        Generar KPIs principales para el dashboard
        """
        kpis = {}
        
        # KPIs básicos
        kpis['cognitive_mean'] = self.df['Cognitive_Score'].mean()
        kpis['cognitive_std'] = self.df['Cognitive_Score'].std()
        kpis['cognitive_p90'] = self.df['Cognitive_Score'].quantile(0.9)
        kpis['cognitive_p10'] = self.df['Cognitive_Score'].quantile(0.1)
        
        # KPIs por demografía
        kpis['gender_performance'] = self.df.groupby('Gender')['Cognitive_Score'].mean().to_dict()
        kpis['diet_performance'] = self.df.groupby('Diet_Type')['Cognitive_Score'].mean().to_dict()
        kpis['exercise_performance'] = self.df.groupby('Exercise_Frequency')['Cognitive_Score'].mean().to_dict()
        
        # Correlaciones importantes
        kpis['sleep_correlation'] = self.df['Sleep_Duration'].corr(self.df['Cognitive_Score'])
        kpis['stress_correlation'] = self.df['Stress_Level'].corr(self.df['Cognitive_Score'])
        kpis['age_correlation'] = self.df['Age'].corr(self.df['Cognitive_Score'])
        
        # Precisión del modelo predictivo
        kpis['prediction_accuracy'] = 1 - np.mean(np.abs(self.df['AI_Predicted_Score'] - self.df['Cognitive_Score']) / self.df['Cognitive_Score'])
        kpis['prediction_r2'] = stats.pearsonr(self.df['AI_Predicted_Score'], self.df['Cognitive_Score'])[0]**2
        
        # Factores de riesgo
        high_screen_time = self.df[self.df['Daily_Screen_Time'] > self.df['Daily_Screen_Time'].quantile(0.75)]
        kpis['high_screen_impact'] = (self.df['Cognitive_Score'].mean() - high_screen_time['Cognitive_Score'].mean())
        
        high_stress = self.df[self.df['Stress_Level'] > self.df['Stress_Level'].quantile(0.75)]
        kpis['high_stress_impact'] = (self.df['Cognitive_Score'].mean() - high_stress['Cognitive_Score'].mean())
        
        return kpis
    
    # =========================
    # FUNCIÓN MAESTRA
    # =========================
    
    def generate_all_charts(self, output_dir='./charts/'):
        """
        Generar todos los gráficos y guardarlos para Power BI
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generando gráficos para Power BI...")
        
        # Gráficos estáticos
        self.plot_cognitive_distribution(f"{output_dir}cognitive_distribution.png", show_plot=False)
        self.plot_age_distribution_by_gender(f"{output_dir}age_by_gender.png", show_plot=False)
        self.plot_cognitive_by_categories(f"{output_dir}cognitive_by_categories.png", show_plot=False)
        self.plot_performance_comparison(f"{output_dir}performance_comparison.png", show_plot=False)
        self.plot_correlation_matrix(f"{output_dir}correlation_matrix.png", show_plot=False)
        self.plot_lifestyle_impact(f"{output_dir}lifestyle_impact.png", show_plot=False)
        
        # Dashboard interactivo
        interactive_fig = self.create_interactive_dashboard(f"{output_dir}interactive_dashboard.html")
        
        # KPIs
        kpis = self.generate_kpis()
        
        # Guardar KPIs en CSV para Power BI
        kpis_df = pd.DataFrame(list(kpis.items()), columns=['KPI', 'Value'])
        kpis_df.to_csv(f"{output_dir}kpis.csv", index=False)
        
        print(f"Gráficos generados en: {output_dir}")
        print("Archivos creados:")
        print("- cognitive_distribution.png")
        print("- age_by_gender.png") 
        print("- cognitive_by_categories.png")
        print("- performance_comparison.png")
        print("- correlation_matrix.png")
        print("- lifestyle_impact.png")
        print("- interactive_dashboard.html")
        print("- kpis.csv")
        
        return kpis

# =========================
# EJEMPLO DE USO
# =========================

"""
# Ejemplo de uso:
# 1. Cargar los datos
df = pd.read_csv('human_cognitive.csv')  # o desde tu base de datos

# 2. Crear instancia de la clase
charts = CognitiveCharts(df)

# 3. Generar todos los gráficos
kpis = charts.generate_all_charts('./power_bi_charts/')

# 4. Ver KPIs principales
print("KPIs Principales:")
for key, value in kpis.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v:.2f}")
    else:
        print(f"{key}: {value:.3f}")

# 5. Generar gráficos individuales si necesitas
charts.plot_cognitive_distribution(save_path='cognitive_dist.png')
charts.plot_correlation_matrix(save_path='correlations.png')

# 6. Para Power BI, también puedes usar:
interactive_dashboard = charts.create_interactive_dashboard()
interactive_dashboard.show()  # Esto abre en el navegador
"""