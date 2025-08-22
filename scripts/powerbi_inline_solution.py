# ====================================
# CÓDIGO PYTHON PARA POWER BI - INLINE
# ====================================
# Copia este código completo en Power BI > Transform Data > Run Python Script

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Power BI usa 'dataset' como nombre automático para tus datos
dataset = pd.read_csv('csv_limpios/human_cognitive_performance.csv/human_cognitive_performance_procesado.csv')
df = dataset

# ====================================
# FUNCIÓN PARA CREAR CARPETA Y GUARDAR
# ====================================

def create_charts_folder(folder_name='charts'):
    """Crear carpeta para gráficos"""
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name

def save_current_chart(chart_name, folder='charts', dpi=300, format='png'):
    """Guardar el gráfico actual"""
    folder_path = create_charts_folder(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{chart_name}_{timestamp}.{format}"
    filepath = os.path.join(folder_path, filename)
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
    print(f"✅ Gráfico guardado: {filepath}")
    return filepath

# ====================================
# CONFIGURACIÓN DE ESTILO
# ====================================
plt.style.use('seaborn-v0_8')
custom_colors = ['#F4A6A6', '#E8989A', '#DC8A8E', '#D07C82', '#C46E76', '#B8606A']
sns.set_palette(custom_colors)

print("🧠 Iniciando análisis cognitivo con guardado automático...")

# ====================================
# GRÁFICO 1: DISTRIBUCIÓN COGNITIVA
# ====================================
print("📊 Generando gráfico 1: Distribución cognitiva...")

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histograma general
# ax1.hist(df['Cognitive_Score'], bins=30, alpha=0.7, 
#         color='#F4A6A6', edgecolor='black')
# ax1.axvline(df['Cognitive_Score'].mean(), color='red', linestyle='--', 
#         label=f'Media: {df["Cognitive_Score"].mean():.2f}')
# ax1.axvline(df['Cognitive_Score'].median(), color='darkred', linestyle='--',
#         label=f'Mediana: {df["Cognitive_Score"].median():.2f}')
# ax1.set_xlabel('Puntuación Cognitiva')
# ax1.set_ylabel('Frecuencia')
# ax1.set_title('Distribución de Puntuaciones Cognitivas')
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # Box plot
# box = ax2.boxplot(df['Cognitive_Score'], vert=True, patch_artist=True)
# box['boxes'][0].set_facecolor('#F4A6A6')
# box['boxes'][0].set_alpha(0.7)
# ax2.set_ylabel('Puntuación Cognitiva')
# ax2.set_title('Box Plot - Puntuaciones Cognitivas')
# ax2.grid(True, alpha=0.3)

# plt.suptitle('Análisis de Distribución Cognitiva', fontsize=16, y=1.02)
# plt.tight_layout()

# # Guardar gráfico
# save_current_chart('cognitive_distribution')
# plt.show()

# ====================================
# GRÁFICO 1.2: DISTRIBUCIÓN
# ====================================

columns_to_plot = ['Age', 'Caffeine_Intake', 'Daily_Screen_Time', 'Diet_Type', 'Gender', 'Sleep_Duration']

def plot_all_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Distribuciónes', fontsize=16, fontweight='bold')
    
    # Aplanar los axes para facilitar la iteración
    axes_flat = axes.flatten()
    
    for i, column in enumerate(columns_to_plot):
        ax = axes_flat[i]
        
        # Verificar si la columna es numérica o categórica
        if df[column].dtype in ['int64', 'float64']:
            # Para variables numéricas: histograma
            ax.hist(df[column], bins=20, alpha=0.7, color='#F4A6A6', edgecolor='black')
            ax.axvline(df[column].mean(), color='red', linestyle='--', 
                    label=f'Media: {df[column].mean():.2f}')
            ax.axvline(df[column].median(), color='darkred', linestyle='--',
                    label=f'Mediana: {df[column].median():.2f}')
            ax.set_ylabel('Frecuencia')
            ax.legend(fontsize=8)
        else:
            # Para variables categóricas: gráfico de barras
            value_counts = df[column].value_counts()
            ax.bar(value_counts.index, value_counts.values, alpha=0.7, 
                color='#A6C8F4', edgecolor='black')
            ax.set_ylabel('Cantidad')
            # Rotar etiquetas si son muy largas
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.set_title(f'Distribución de {column.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
    
    # Ocultar el último subplot si no se usa
    if len(columns_to_plot) < len(axes_flat):
        axes_flat[-1].set_visible(False)
    
    plt.tight_layout()
    plt.show()

plot_all_distributions(df)

# ====================================
# GRÁFICO 2: MATRIZ DE CORRELACIÓN
# ====================================
print("📊 Generando gráfico 2: Matriz de correlaciones...")

numeric_cols = ['Age', 'AI_Predicted_Score', 'Caffeine_Intake', 
            'Cognitive_Score', 'Daily_Screen_Time', 'Memory_Test_Score',
            'Reaction_Time', 'Sleep_Duration', 'Stress_Level']

# Verificar que las columnas existen en el dataset
available_cols = [col for col in numeric_cols if col in df.columns]
print(f"Columnas disponibles para correlación: {available_cols}")

corr_matrix = df[available_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))

# Crear máscara para la diagonal superior
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Heatmap
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
        square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)

ax.set_title('Matriz de Correlación - Variables Cognitivas', fontsize=16, pad=20)
plt.tight_layout()

# Guardar gráfico
save_current_chart('correlation_matrix')
plt.show()

# ====================================
# GRÁFICO 3: ANÁLISIS DEMOGRÁFICO
# ====================================
print("📊 Generando gráfico 3: Análisis demográfico...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Por género (si existe la columna)
if 'Gender' in df.columns:
    sns.boxplot(data=df, x='Gender', y='Cognitive_Score', ax=axes[0,0])
    axes[0,0].set_title('Puntuación Cognitiva por Género')
    axes[0,0].grid(True, alpha=0.3)
else:
    axes[0,0].text(0.5, 0.5, 'Columna Gender no disponible', 
                ha='center', va='center', transform=axes[0,0].transAxes)
    axes[0,0].set_title('Género - No disponible')

# Por tipo de dieta (si existe la columna)
if 'Diet_Type' in df.columns:
    sns.boxplot(data=df, x='Diet_Type', y='Cognitive_Score', ax=axes[0,1])
    axes[0,1].set_title('Puntuación Cognitiva por Tipo de Dieta')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
else:
    axes[0,1].text(0.5, 0.5, 'Columna Diet_Type no disponible', 
                ha='center', va='center', transform=axes[0,1].transAxes)
    axes[0,1].set_title('Dieta - No disponible')

# Por frecuencia de ejercicio (si existe la columna)
if 'Exercise_Frequency' in df.columns:
    sns.boxplot(data=df, x='Exercise_Frequency', y='Cognitive_Score', ax=axes[1,0])
    axes[1,0].set_title('Puntuación Cognitiva por Frecuencia de Ejercicio')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
else:
    axes[1,0].text(0.5, 0.5, 'Columna Exercise_Frequency no disponible', 
                ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Ejercicio - No disponible')

# Por grupos de edad
if 'Age' in df.columns:
    age_groups = pd.cut(df['Age'], bins=[0, 25, 40, 100], 
                    labels=['18-25', '25-40', '40+'])
    temp_df = df.copy()
    temp_df['Age_Group'] = age_groups
    sns.boxplot(data=temp_df, x='Age_Group', y='Cognitive_Score', ax=axes[1,1])
    axes[1,1].set_title('Puntuación Cognitiva por Grupo de Edad')
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, 'Columna Age no disponible', 
                ha='center', va='center', transform=axes[1,1].transAxes)
    axes[1,1].set_title('Edad - No disponible')

plt.suptitle('Análisis Demográfico Completo', fontsize=16, y=1.02)
plt.tight_layout()

# Guardar gráfico
save_current_chart('demographic_analysis')
plt.show()

# ====================================
# GRÁFICO 4: FACTORES DE ESTILO DE VIDA
# ====================================
print("📊 Generando gráfico 4: Factores de estilo de vida...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Screen Time vs Cognitive Score
if 'Daily_Screen_Time' in df.columns:
    screen_bins = pd.cut(df['Daily_Screen_Time'], bins=4, 
                    labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
    screen_data = df.groupby(screen_bins)['Cognitive_Score'].mean()
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
else:
    axes[0,0].text(0.5, 0.5, 'Columna Daily_Screen_Time no disponible', 
                ha='center', va='center', transform=axes[0,0].transAxes)
    axes[0,0].set_title('Tiempo de Pantalla - No disponible')

# Stress Level vs Cognitive Score
if 'Stress_Level' in df.columns:
    stress_bins = pd.cut(df['Stress_Level'], bins=3, 
                    labels=['Bajo', 'Medio', 'Alto'])
    stress_data = df.groupby(stress_bins)['Cognitive_Score'].mean()
    bars2 = axes[0,1].bar(stress_data.index, stress_data.values, 
                        color='#E8989A', alpha=0.7)
    axes[0,1].set_xlabel('Nivel de Estrés')
    axes[0,1].set_ylabel('Puntuación Cognitiva Promedio')
    axes[0,1].set_title('Impacto del Estrés')
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, stress_data.values):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
else:
    axes[0,1].text(0.5, 0.5, 'Columna Stress_Level no disponible', 
                ha='center', va='center', transform=axes[0,1].transAxes)
    axes[0,1].set_title('Estrés - No disponible')

# Sleep Duration scatter
# if 'Sleep_Duration' in df.columns:
#     axes[1,0].scatter(df['Sleep_Duration'], df['Cognitive_Score'], 
#                     alpha=0.6, color='#DC8A8E')
#     z = np.polyfit(df['Sleep_Duration'], df['Cognitive_Score'], 1)
#     p = np.poly1d(z)
#     axes[1,0].plot(df['Sleep_Duration'], p(df['Sleep_Duration']), 
#                 "r--", alpha=0.8)
#     axes[1,0].set_xlabel('Duración del Sueño (horas)')
#     axes[1,0].set_ylabel('Puntuación Cognitiva')
#     axes[1,0].set_title('Sueño vs Rendimiento Cognitivo')
#     axes[1,0].grid(True, alpha=0.3)
# else:
#     axes[1,0].text(0.5, 0.5, 'Columna Sleep_Duration no disponible', 
#                 ha='center', va='center', transform=axes[1,0].transAxes)
#     axes[1,0].set_title('Sueño - No disponible')

def create_grouped_scatterplot(df, x_col, y_col, ax, bins=None, labels=None):
    """
    Función para crear scatterplot agrupado
    
    Parameters:
    - df: DataFrame
    - x_col: columna para eje X
    - y_col: columna para eje Y (será agrupada)
    - ax: eje de matplotlib
    - bins: rangos para agrupar (default: [0, 25, 50, 75, 100])
    - labels: etiquetas para los grupos
    """
    
    if bins is None:
        bins = [0, 25, 50, 75, 100]
    if labels is None:
        labels = ['0-25', '26-50', '51-75', '76-100']
    
    # Crear grupos
    df[f'{y_col}_Group'] = pd.cut(df[y_col], bins=bins, labels=labels, include_lowest=True)
    
    # Colores para cada grupo
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#BB8FCE']
    
    # Crear scatter plot
    for i, (group, color) in enumerate(zip(labels, colors[:len(labels)])):
        group_data = df[df[f'{y_col}_Group'] == group]
        if not group_data.empty:
            ax.scatter(group_data[x_col], group_data[y_col], 
                    alpha=0.7, color=color, label=f'{y_col} {group}', 
                    s=60, edgecolors='white', linewidth=0.5)

    # Línea de tendencia
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2, 
            label='Tendencia general')
    
    # Líneas horizontales para los rangos
    for threshold in bins[1:-1]:  # Excluir el primer y último valor
        ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.4)
    
    # Configuración
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Estadísticas por grupo
    print(f"\nEstadísticas por grupo para {y_col}:")
    for group in labels:
        group_data = df[df[f'{y_col}_Group'] == group]
        if not group_data.empty:
            print(f"{group}: {len(group_data)} observaciones")
            print(f"  Media {x_col}: {group_data[x_col].mean():.2f}")
            print(f"  Media {y_col}: {group_data[y_col].mean():.2f}")

create_grouped_scatterplot(df, 'Cognitive_Score', 'Sleep_Duration', axes[1,0],
                        bins=[0, 20, 40, 60, 80, 100],
                        labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])

# Age vs Reaction Time
# if 'Age' in df.columns and 'Reaction_Time' in df.columns:
#     axes[1,1].scatter(df['Age'], df['Reaction_Time'], 
#                     alpha=0.6, color='#D07C82')
#     z = np.polyfit(df['Age'], df['Reaction_Time'], 1)
#     p = np.poly1d(z)
#     axes[1,1].plot(df['Age'], p(df['Age']), "r--", alpha=0.8)
#     axes[1,1].set_xlabel('Edad')
#     axes[1,1].set_ylabel('Tiempo de Reacción')
#     axes[1,1].set_title('Edad vs Tiempo de Reacción')
#     axes[1,1].grid(True, alpha=0.3)
# else:
#     axes[1,1].text(0.5, 0.5, 'Columnas Age/Reaction_Time no disponibles', 
#                 ha='center', va='center', transform=axes[1,1].transAxes)
#     axes[1,1].set_title('Edad/Reacción - No disponible')

# plt.suptitle('Análisis de Factores de Estilo de Vida', fontsize=16, y=1.02)
# plt.tight_layout()

def create_x_grouped_scatterplot(df, x_col, y_col, ax, group_size=25, custom_bins=None, custom_labels=None):
    """
    Función para crear scatterplot agrupado por variable X
    
    Parameters:
    - df: DataFrame
    - x_col: columna para eje X (será agrupada)
    - y_col: columna para eje Y
    - ax: eje de matplotlib
    - group_size: tamaño del grupo para variable continua (default: 25)
    - custom_bins: bins personalizados
    - custom_labels: etiquetas personalizadas
    """
    
    if custom_bins is None:
        # Crear bins automáticamente
        min_val = df[x_col].min()
        max_val = df[x_col].max()
        
        start_val = (min_val // group_size) * group_size
        end_val = ((max_val // group_size) + 1) * group_size
        bins = list(range(int(start_val), int(end_val) + 1, group_size))
        
        # Si no hay suficientes bins, usar cuartiles
        if len(bins) <= 2:
            bins = [min_val, df[x_col].quantile(0.25), df[x_col].quantile(0.5), 
                    df[x_col].quantile(0.75), max_val]
    else:
        bins = custom_bins
    
    if custom_labels is None:
        if len(bins) > 2:
            labels = [f'{int(bins[i])}-{int(bins[i+1]-1)}' for i in range(len(bins)-1)]
            labels[-1] = f'{int(bins[-2])}-{int(max_val)}'
        else:
            labels = [f'Grupo {i+1}' for i in range(len(bins)-1)]
    else:
        labels = custom_labels
    
    # Crear grupos
    df[f'{x_col}_Group'] = pd.cut(df[x_col], bins=bins, labels=labels, include_lowest=True)
    
    # Colores
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#BB8FCE', '#F1948A', '#85C1E9']
    
    # Scatter plot
    for i, (group, color) in enumerate(zip(labels, colors[:len(labels)])):
        group_data = df[df[f'{x_col}_Group'] == group]
        if not group_data.empty:
            ax.scatter(group_data[x_col], group_data[y_col], 
                    alpha=0.7, color=color, label=f'{x_col} {group}', 
                    s=60, edgecolors='white', linewidth=0.5)
    
    # Línea de tendencia
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2, 
            label='Tendencia general')
    
    # Líneas verticales de referencia
    for threshold in bins[1:-1]:
        ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.4)
    
    # Configuración
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()} (Agrupado)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Estadísticas por grupo
    print(f"\nEstadísticas por grupo para {x_col}:")
    for group in labels:
        group_data = df[df[f'{x_col}_Group'] == group]
        if not group_data.empty:
            print(f"{group}: {len(group_data)} observaciones")
            print(f"  Media {x_col}: {group_data[x_col].mean():.2f}")
            print(f"  Media {y_col}: {group_data[y_col].mean():.2f}")

create_x_grouped_scatterplot(df, 'Reaction_Time', 'Cognitive_Score', axes[1,1],
                            custom_bins=[18, 30, 45, 60, 80],
                            custom_labels=['Jóvenes', 'Adultos', 'Mediana Edad', 'Mayores'])

# Guardar gráfico
save_current_chart('lifestyle_factors')
plt.show()

# ====================================
# GRÁFICO 5: RENDIMIENTO DEL MODELO
# ====================================
print("📊 Generando gráfico 5: Rendimiento del modelo...")



# ====================================
# CALCULAR KPIs
# ====================================
print("📊 Calculando KPIs principales...")

kpis_data = {
    'KPI_Name': [],
    'KPI_Value': [],
    'KPI_Category': []
}

# KPIs básicos
kpis_data['KPI_Name'].extend(['Cognitive_Mean', 'Cognitive_Std', 'Cognitive_Median'])
kpis_data['KPI_Value'].extend([
    df['Cognitive_Score'].mean(),
    df['Cognitive_Score'].std(),
    df['Cognitive_Score'].median()
])
kpis_data['KPI_Category'].extend(['Basic', 'Basic', 'Basic'])

# Correlaciones (si las columnas existen)
for col in ['Sleep_Duration', 'Stress_Level', 'Age']:
    if col in df.columns:
        corr_value = df[col].corr(df['Cognitive_Score'])
        kpis_data['KPI_Name'].append(f'{col}_Correlation')
        kpis_data['KPI_Value'].append(corr_value)
        kpis_data['KPI_Category'].append('Correlation')

# KPIs del modelo (si disponibles)
if 'AI_Predicted_Score' in df.columns:
    mae = np.mean(np.abs(df['AI_Predicted_Score'] - df['Cognitive_Score']))
    r2 = np.corrcoef(df['AI_Predicted_Score'], df['Cognitive_Score'])[0,1]**2
    
    kpis_data['KPI_Name'].extend(['Model_MAE', 'Model_R2'])
    kpis_data['KPI_Value'].extend([mae, r2])
    kpis_data['KPI_Category'].extend(['Model', 'Model'])

# Crear DataFrame de KPIs
kpis_df = pd.DataFrame(kpis_data)

# Mostrar KPIs
print("\n" + "="*50)
print("📈 KPIs CALCULADOS:")
print("="*50)
for _, row in kpis_df.iterrows():
    print(f"{row['KPI_Name']}: {row['KPI_Value']:.3f}")

print(f"\n✅ ANÁLISIS COMPLETADO - {datetime.now().strftime('%H:%M:%S')}")
print("📁 Todos los gráficos guardados en la carpeta 'charts/'")
print("🔗 Listos para usar en Power BI")

# El DataFrame kpis_df estará disponible para Power BI
