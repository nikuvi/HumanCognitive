import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ChartSaver:
    """
    Clase para guardar gráficos automáticamente en carpeta charts/
    """
    
    def __init__(self, charts_folder='charts'):
        """
        Inicializar el guardador de gráficos
        
        Args:
            charts_folder (str): Nombre de la carpeta donde guardar los gráficos
        """
        self.charts_folder = charts_folder
        self.create_charts_folder()
        
    def create_charts_folder(self):
        """
        Crear la carpeta de gráficos si no existe
        """
        Path(self.charts_folder).mkdir(parents=True, exist_ok=True)
        print(f"📁 Carpeta '{self.charts_folder}' lista para guardar gráficos")
    
    def save_chart(self, fig=None, chart_name='chart', 
                   subtitle='', dpi=300, format='png', 
                   bbox_inches='tight', show_timestamp=True):
        """
        Guardar un gráfico individual en la carpeta charts/
        
        Args:
            fig: Figura de matplotlib (si None, usa plt.gcf())
            chart_name (str): Nombre base del archivo
            subtitle (str): Subtítulo para el nombre del archivo
            dpi (int): Resolución de la imagen
            format (str): Formato del archivo ('png', 'jpg', 'svg', 'pdf')
            bbox_inches (str): Ajuste de bordes
            show_timestamp (bool): Incluir timestamp en el nombre
        
        Returns:
            str: Ruta completa del archivo guardado
        """
        if fig is None:
            fig = plt.gcf()
        
        # Crear nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if show_timestamp else ""
        subtitle_part = f"_{subtitle}" if subtitle else ""
        timestamp_part = f"_{timestamp}" if timestamp else ""
        
        filename = f"{chart_name}{subtitle_part}{timestamp_part}.{format}"
        filepath = os.path.join(self.charts_folder, filename)
        
        # Guardar el gráfico
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   format=format, facecolor='white', edgecolor='none')
        
        print(f"✅ Gráfico guardado: {filepath}")
        return filepath
    
    def save_multiple_formats(self, fig=None, chart_name='chart', 
                             formats=['png', 'svg'], **kwargs):
        """
        Guardar el mismo gráfico en múltiples formatos
        
        Args:
            fig: Figura de matplotlib
            chart_name (str): Nombre base del archivo
            formats (list): Lista de formatos a guardar
            **kwargs: Argumentos adicionales para save_chart
        
        Returns:
            list: Lista de rutas de archivos guardados
        """
        saved_files = []
        for format_type in formats:
            filepath = self.save_chart(fig=fig, chart_name=chart_name, 
                                     format=format_type, **kwargs)
            saved_files.append(filepath)
        return saved_files

# =================================================================
# CLASE PRINCIPAL CON FUNCIONES DE GUARDADO INTEGRADAS
# =================================================================

class CognitiveChartsWithSaver:
    """
    Clase mejorada para análisis cognitivo con guardado automático de gráficos
    """
    
    def __init__(self, df, charts_folder='charts'):
        """
        Inicializar con el DataFrame y configurar guardado automático
        
        Args:
            df (pandas.DataFrame): DataFrame con los datos
            charts_folder (str): Carpeta donde guardar los gráficos
        """
        self.df = df
        self.saver = ChartSaver(charts_folder)
        self.numeric_cols = ['Age', 'AI_Predicted_Score', 'Caffeine_Intake', 
                            'Cognitive_Score', 'Daily_Screen_Time', 'Memory_Test_Score',
                            'Reaction_Time', 'Sleep_Duration', 'Stress_Level']
        self.categorical_cols = ['Diet_Type', 'Exercise_Frequency', 'Gender']
        
        # Configurar estilo personalizado
        plt.style.use('seaborn-v0_8')
        custom_colors = ['#F4A6A6', '#E8989A', '#DC8A8E', '#D07C82', '#C46E76', '#B8606A']
        sns.set_palette(custom_colors)
        
        print("🧠 Análisis Cognitivo inicializado con guardado automático")
    
    def plot_cognitive_distribution(self, save=True, show=True):
        """
        Distribución de puntuaciones cognitivas con guardado automático
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma
        ax1.hist(self.df['Cognitive_Score'], bins=30, alpha=0.7, 
                color='#F4A6A6', edgecolor='black')
        ax1.axvline(self.df['Cognitive_Score'].mean(), color='red', linestyle='--', 
                   label=f'Media: {self.df["Cognitive_Score"].mean():.2f}')
        ax1.axvline(self.df['Cognitive_Score'].median(), color='darkred', linestyle='--',
                   label=f'Mediana: {self.df["Cognitive_Score"].median():.2f}')
        ax1.set_xlabel('Puntuación Cognitiva')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Puntuaciones Cognitivas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        box = ax2.boxplot(self.df['Cognitive_Score'], vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('#F4A6A6')
        box['boxes'][0].set_alpha(0.7)
        ax2.set_ylabel('Puntuación Cognitiva')
        ax2.set_title('Box Plot - Puntuaciones Cognitivas')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Análisis de Distribución Cognitiva', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            self.saver.save_chart(fig, 'cognitive_distribution', 
                                 subtitle='histogram_boxplot')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_correlation_matrix(self, save=True, show=True):
        """
        Matriz de correlación con guardado automático
        """
        corr_matrix = self.df[self.numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Crear máscara para la diagonal superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Matriz de Correlación - Variables Cognitivas', 
                    fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save:
            self.saver.save_chart(fig, 'correlation_matrix', 
                                 subtitle='cognitive_variables')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_demographic_analysis(self, save=True, show=True):
        """
        Análisis demográfico completo con guardado automático
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
        age_groups = pd.cut(self.df['Age'], bins=[0, 30, 50, 100], 
                           labels=['18-30', '31-50', '50+'])
        temp_df = self.df.copy()
        temp_df['Age_Group'] = age_groups
        sns.boxplot(data=temp_df, x='Age_Group', y='Cognitive_Score', ax=axes[1,1])
        axes[1,1].set_title('Puntuación Cognitiva por Grupo de Edad')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Análisis Demográfico Completo', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            self.saver.save_chart(fig, 'demographic_analysis', 
                                 subtitle='by_categories')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_lifestyle_factors(self, save=True, show=True):
        """
        Análisis de factores de estilo de vida con guardado automático
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Screen Time vs Cognitive Score
        screen_bins = pd.cut(self.df['Daily_Screen_Time'], bins=4, 
                           labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
        screen_data = self.df.groupby(screen_bins)['Cognitive_Score'].mean()
        bars1 = axes[0,0].bar(screen_data.index, screen_data.values, 
                             color='#F4A6A6', alpha=0.7)
        axes[0,0].set_xlabel('Tiempo de Pantalla Diario')
        axes[0,0].set_ylabel('Puntuación Cognitiva Promedio')
        axes[0,0].set_title('Impacto del Tiempo de Pantalla')
        axes[0,0].grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, value in zip(bars1, screen_data.values):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{value:.1f}', ha='center', va='bottom')
        
        # Stress Level vs Cognitive Score
        stress_bins = pd.cut(self.df['Stress_Level'], bins=3, 
                           labels=['Bajo', 'Medio', 'Alto'])
        stress_data = self.df.groupby(stress_bins)['Cognitive_Score'].mean()
        bars2 = axes[0,1].bar(stress_data.index, stress_data.values, 
                             color='#E8989A', alpha=0.7)
        axes[0,1].set_xlabel('Nivel de Estrés')
        axes[0,1].set_ylabel('Puntuación Cognitiva Promedio')
        axes[0,1].set_title('Impacto del Estrés')
        axes[0,1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, stress_data.values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{value:.1f}', ha='center', va='bottom')
        
        # Sleep Duration scatter
        axes[1,0].scatter(self.df['Sleep_Duration'], self.df['Cognitive_Score'], 
                         alpha=0.6, color='#DC8A8E')
        z = np.polyfit(self.df['Sleep_Duration'], self.df['Cognitive_Score'], 1)
        p = np.poly1d(z)
        axes[1,0].plot(self.df['Sleep_Duration'], p(self.df['Sleep_Duration']), 
                      "r--", alpha=0.8)
        axes[1,0].set_xlabel('Duración del Sueño (horas)')
        axes[1,0].set_ylabel('Puntuación Cognitiva')
        axes[1,0].set_title('Sueño vs Rendimiento Cognitivo')
        axes[1,0].grid(True, alpha=0.3)
        
        # Age vs Reaction Time
        axes[1,1].scatter(self.df['Age'], self.df['Reaction_Time'], 
                         alpha=0.6, color='#D07C82')
        z = np.polyfit(self.df['Age'], self.df['Reaction_Time'], 1)
        p = np.poly1d(z)
        axes[1,1].plot(self.df['Age'], p(self.df['Age']), "r--", alpha=0.8)
        axes[1,1].set_xlabel('Edad')
        axes[1,1].set_ylabel('Tiempo de Reacción')
        axes[1,1].set_title('Edad vs Tiempo de Reacción')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Análisis de Factores de Estilo de Vida', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            self.saver.save_chart(fig, 'lifestyle_factors', 
                                 subtitle='comprehensive_analysis')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_model_performance(self, save=True, show=True):
        """
        Análisis del rendimiento del modelo predictivo con guardado automático
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Predicted vs Actual
        axes[0,0].scatter(self.df['AI_Predicted_Score'], self.df['Cognitive_Score'], 
                         alpha=0.6, color='#F4A6A6')
        min_val = min(self.df['AI_Predicted_Score'].min(), self.df['Cognitive_Score'].min())
        max_val = max(self.df['AI_Predicted_Score'].max(), self.df['Cognitive_Score'].max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                      label='Predicción Perfecta')
        axes[0,0].set_xlabel('AI Predicted Score')
        axes[0,0].set_ylabel('Cognitive Score Real')
        axes[0,0].set_title('Predicción vs Realidad')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Residuales
        residuals = self.df['Cognitive_Score'] - self.df['AI_Predicted_Score']
        axes[0,1].scatter(self.df['AI_Predicted_Score'], residuals, alpha=0.6, color='#E8989A')
        axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0,1].set_xlabel('AI Predicted Score')
        axes[0,1].set_ylabel('Residuales (Real - Predicho)')
        axes[0,1].set_title('Análisis de Residuales')
        axes[0,1].grid(True, alpha=0.3)
        
        # Distribución de residuales
        axes[1,0].hist(residuals, bins=30, alpha=0.7, color='#DC8A8E', edgecolor='black')
        axes[1,0].axvline(residuals.mean(), color='red', linestyle='--', 
                         label=f'Media: {residuals.mean():.2f}')
        axes[1,0].set_xlabel('Residuales')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].set_title('Distribución de Residuales')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Métricas del modelo
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = np.corrcoef(self.df['AI_Predicted_Score'], self.df['Cognitive_Score'])[0,1]**2
        
        # Texto con métricas
        axes[1,1].text(0.1, 0.8, f'MAE: {mae:.3f}', fontsize=14, transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.7, f'RMSE: {rmse:.3f}', fontsize=14, transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.6, f'R²: {r2:.3f}', fontsize=14, transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.5, f'Precisión: {(1-mae/self.df["Cognitive_Score"].mean())*100:.1f}%', 
                      fontsize=14, transform=axes[1,1].transAxes)
        axes[1,1].set_title('Métricas del Modelo')
        axes[1,1].axis('off')
        
        plt.suptitle('Análisis del Rendimiento del Modelo Predictivo', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            self.saver.save_chart(fig, 'model_performance', 
                                 subtitle='predictive_analysis')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def generate_all_charts(self, save=True, show=False, formats=['png']):
        """
        Generar todos los gráficos y guardarlos automáticamente
        
        Args:
            save (bool): Guardar los gráficos
            show (bool): Mostrar los gráficos
            formats (list): Formatos en los que guardar ('png', 'svg', 'pdf', 'jpg')
        
        Returns:
            dict: Diccionario con las figuras generadas
        """
        print("📊 GENERANDO TODOS LOS GRÁFICOS...")
        print("=" * 50)
        
        figures = {}
        
        # 1. Distribución cognitiva
        print("🧠 1. Análisis de distribución cognitiva...")
        figures['distribution'] = self.plot_cognitive_distribution(save=False, show=show)
        if save:
            for fmt in formats:
                self.saver.save_chart(figures['distribution'], 
                                     'cognitive_distribution', format=fmt)
        
        # 2. Matriz de correlación
        print("🌐 2. Matriz de correlaciones...")
        figures['correlation'] = self.plot_correlation_matrix(save=False, show=show)
        if save:
            for fmt in formats:
                self.saver.save_chart(figures['correlation'], 
                                     'correlation_matrix', format=fmt)
        
        # 3. Análisis demográfico
        print("👥 3. Análisis demográfico...")
        figures['demographic'] = self.plot_demographic_analysis(save=False, show=show)
        if save:
            for fmt in formats:
                self.saver.save_chart(figures['demographic'], 
                                     'demographic_analysis', format=fmt)
        
        # 4. Factores de estilo de vida
        print("🏃 4. Factores de estilo de vida...")
        figures['lifestyle'] = self.plot_lifestyle_factors(save=False, show=show)
        if save:
            for fmt in formats:
                self.saver.save_chart(figures['lifestyle'], 
                                     'lifestyle_factors', format=fmt)
        
        # 5. Rendimiento del modelo
        print("🤖 5. Rendimiento del modelo...")
        figures['model'] = self.plot_model_performance(save=False, show=show)
        if save:
            for fmt in formats:
                self.saver.save_chart(figures['model'], 
                                     'model_performance', format=fmt)
        
        # Cerrar figuras si no se muestran
        if not show:
            for fig in figures.values():
                plt.close(fig)
        
        print(f"\n✅ TODOS LOS GRÁFICOS GENERADOS Y GUARDADOS EN '{self.saver.charts_folder}/'")
        print(f"📊 Total de gráficos: {len(figures)} x {len(formats)} formatos")
        
        return figures

# =================================================================
# EJEMPLO DE USO COMPLETO
# =================================================================

"""
# Ejemplo de uso completo:

# 1. Cargar datos
df = pd.read_csv('human_cognitive.csv')

# 2. Crear instancia con guardado automático
charts = CognitiveChartsWithSaver(df, charts_folder='charts')

# 3. Generar gráficos individuales (se guardan automáticamente)
charts.plot_cognitive_distribution()
charts.plot_correlation_matrix()
charts.plot_demographic_analysis()

# 4. Generar todos los gráficos de una vez en múltiples formatos
figures = charts.generate_all_charts(
    save=True, 
    show=False,  # No mostrar para generación masiva
    formats=['png', 'svg']  # PNG para Power BI, SVG para web
)

# 5. También puedes guardar un gráfico específico en formatos personalizados
fig = charts.plot_lifestyle_factors(save=False, show=False)
charts.saver.save_multiple_formats(
    fig, 
    'lifestyle_custom', 
    formats=['png', 'pdf', 'svg'],
    dpi=600  # Alta resolución para publicación
)

print("¡Todos los gráficos guardados en la carpeta charts/!")
"""
