"""
Sistema de Visualización de Simulación de Llamadas
===================================================

Módulo para crear visualizaciones animadas y comparativas del proceso
de búsqueda de contactos usando diferentes estrategias.

Autor: Sistema de ML para Cobranzas
Fecha: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from IPython.display import HTML
import pandas as pd


class ContactAnimator:
    """
    Clase para crear animaciones del proceso de llamadas.
    Muestra visualmente cómo cada estrategia va encontrando contactos.
    """

    def __init__(self, resultados_simulacion, fps=20, duracion_segundos=20):
        """
        Args:
            resultados_simulacion (dict): Resultados de simulación por estrategia
            fps (int): Frames por segundo de la animación
            duracion_segundos (int): Duración total de la animación
        """
        self.resultados = resultados_simulacion
        self.fps = fps
        self.duracion = duracion_segundos
        self.total_frames = fps * duracion_segundos

        # Obtener total de intentos
        first_result = next(iter(resultados_simulacion.values()))
        self.total_intentos = len(first_result['df_simulacion'])
        self.total_contactos = first_result['df_simulacion']['contacto_real'].sum()

    def crear_animacion_comparativa(self, guardar_como=None):
        """
        Crea animación comparativa de las 4 estrategias.

        Args:
            guardar_como (str, optional): Nombre de archivo para guardar (ej: 'animacion.mp4')

        Returns:
            Animation object para mostrar en Jupyter
        """
        print("\n🎬 Creando animación comparativa...")

        estrategias = list(self.resultados.keys())
        n_estrategias = len(estrategias)

        # Configurar figura
        fig, axes = plt.subplots(n_estrategias, 2, figsize=(16, 4*n_estrategias))
        if n_estrategias == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Simulación de Estrategias de Llamadas en Cobranza',
                     fontsize=16, fontweight='bold', y=0.995)

        # Configurar cada fila (una por estrategia)
        artists = []
        for i, estrategia in enumerate(estrategias):
            resultado = self.resultados[estrategia]
            df_sim = resultado['df_simulacion']

            # Gráfico izquierdo: Progreso de contactos encontrados
            ax_progress = axes[i, 0]
            ax_progress.set_xlim(0, 100)
            ax_progress.set_ylim(0, 100)
            ax_progress.set_xlabel('% del Dataset Recorrido')
            ax_progress.set_ylabel('% de Contactos Encontrados')
            ax_progress.set_title(f'{estrategia}', fontweight='bold', fontsize=12)
            ax_progress.grid(alpha=0.3)

            # Línea diagonal (aleatorio perfecto)
            ax_progress.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Aleatorio ideal')

            # Línea de progreso (se actualizará)
            line, = ax_progress.plot([], [], 'b-', linewidth=2, label='Progreso real')

            # Marcadores de hitos
            for hito in [50, 80, 95]:
                ax_progress.axhline(y=hito, color='gray', linestyle=':', alpha=0.2)
                ax_progress.text(102, hito, f'{hito}%', fontsize=8, va='center')

            ax_progress.legend(loc='upper left', fontsize=8)

            # Gráfico derecho: Métricas en tiempo real
            ax_metrics = axes[i, 1]
            ax_metrics.axis('off')

            # Textos que se actualizarán
            text_llamadas = ax_metrics.text(0.1, 0.9, '', fontsize=11, transform=ax_metrics.transAxes)
            text_contactos = ax_metrics.text(0.1, 0.75, '', fontsize=11, transform=ax_metrics.transAxes)
            text_tasa = ax_metrics.text(0.1, 0.6, '', fontsize=11, transform=ax_metrics.transAxes)
            text_progreso = ax_metrics.text(0.1, 0.45, '', fontsize=11, transform=ax_metrics.transAxes,
                                          fontweight='bold', color='blue')

            # Etiqueta de la barra
            text_barra_label = ax_metrics.text(0.1, 0.38, 'Progreso de contactos:',
                                              fontsize=9, transform=ax_metrics.transAxes,
                                              style='italic', color='gray')

            # Barra de progreso visual (representa % de CONTACTOS encontrados)
            rect_bg = Rectangle((0.1, 0.25), 0.8, 0.1, transform=ax_metrics.transAxes,
                               facecolor='lightgray', edgecolor='black')
            ax_metrics.add_patch(rect_bg)

            rect_progress = Rectangle((0.1, 0.25), 0, 0.1, transform=ax_metrics.transAxes,
                                     facecolor='orange', edgecolor='black')
            ax_metrics.add_patch(rect_progress)

            # Texto de porcentaje sobre la barra
            text_barra_pct = ax_metrics.text(0.5, 0.3, '0%', fontsize=10,
                                            transform=ax_metrics.transAxes,
                                            ha='center', va='center', fontweight='bold')

            # Guardar referencias para actualización
            artists.append({
                'line': line,
                'text_llamadas': text_llamadas,
                'text_contactos': text_contactos,
                'text_tasa': text_tasa,
                'text_progreso': text_progreso,
                'rect_progress': rect_progress,
                'text_barra_pct': text_barra_pct,
                'df_sim': df_sim
            })

        # Función de inicialización
        def init():
            for artist in artists:
                artist['line'].set_data([], [])
            return []

        # Función de actualización por frame
        def update(frame):
            # Calcular qué porcentaje del dataset hemos recorrido
            progreso = (frame / self.total_frames)
            idx = int(progreso * self.total_intentos)
            if idx >= self.total_intentos:
                idx = self.total_intentos - 1

            updated_artists = []

            for artist in artists:
                df_sim = artist['df_sim']

                # Obtener datos hasta este punto
                df_hasta_ahora = df_sim.iloc[:idx+1]

                if len(df_hasta_ahora) > 0:
                    # Actualizar línea de progreso
                    x_data = df_hasta_ahora['porcentaje_llamadas'].values
                    y_data = df_hasta_ahora['porcentaje_contactos'].values
                    artist['line'].set_data(x_data, y_data)

                    # Obtener métricas actuales
                    llamadas_actuales = len(df_hasta_ahora)
                    contactos_actuales = df_hasta_ahora['contacto_real'].sum()
                    pct_contactos = (contactos_actuales / self.total_contactos) * 100
                    tasa_exito = (contactos_actuales / llamadas_actuales) * 100 if llamadas_actuales > 0 else 0

                    # Actualizar textos
                    artist['text_llamadas'].set_text(f'Llamadas realizadas: {llamadas_actuales:,}')
                    artist['text_contactos'].set_text(f'Contactos encontrados: {contactos_actuales:,} / {self.total_contactos:,}')
                    artist['text_tasa'].set_text(f'Tasa de éxito: {tasa_exito:.2f}%')

                    # Determinar estado actual
                    if pct_contactos >= 99:
                        status = '✓ COMPLETADO'
                        color = 'green'
                    elif pct_contactos >= 80:
                        status = f'🎯 {pct_contactos:.1f}% encontrados'
                        color = 'blue'
                    else:
                        status = f'⏳ {pct_contactos:.1f}% encontrados'
                        color = 'orange'

                    artist['text_progreso'].set_text(status)
                    artist['text_progreso'].set_color(color)

                    # Actualizar barra de progreso basada en % de CONTACTOS encontrados
                    # No en % del dataset recorrido
                    barra_width = min((pct_contactos / 100.0) * 0.8, 0.8)  # Max 0.8 (80% del ancho)
                    artist['rect_progress'].set_width(barra_width)

                    # Cambiar color de barra según progreso (gradiente de 13 colores)
                    if pct_contactos >= 99.5:
                        artist['rect_progress'].set_facecolor('#006400')  # Verde oscuro - COMPLETADO
                    elif pct_contactos >= 99:
                        artist['rect_progress'].set_facecolor('#00AA00')  # Verde - 99%
                    elif pct_contactos >= 95:
                        artist['rect_progress'].set_facecolor('#33CC33')  # Verde claro - 95%
                    elif pct_contactos >= 90:
                        artist['rect_progress'].set_facecolor('#66DD66')  # Verde muy claro - 90%
                    elif pct_contactos >= 80:
                        artist['rect_progress'].set_facecolor('#99FF33')  # Verde-amarillo - 80%
                    elif pct_contactos >= 70:
                        artist['rect_progress'].set_facecolor('#CCFF00')  # Amarillo-verde - 70%
                    elif pct_contactos >= 60:
                        artist['rect_progress'].set_facecolor('#FFFF00')  # Amarillo puro - 60%
                    elif pct_contactos >= 50:
                        artist['rect_progress'].set_facecolor('#FFCC00')  # Amarillo-naranja - 50%
                    elif pct_contactos >= 40:
                        artist['rect_progress'].set_facecolor('#FF9900')  # Naranja claro - 40%
                    elif pct_contactos >= 30:
                        artist['rect_progress'].set_facecolor('#FF6600')  # Naranja - 30%
                    elif pct_contactos >= 20:
                        artist['rect_progress'].set_facecolor('#FF3300')  # Rojo-naranja - 20%
                    elif pct_contactos >= 10:
                        artist['rect_progress'].set_facecolor('#CC0000')  # Rojo - 10%
                    else:
                        artist['rect_progress'].set_facecolor('#880000')  # Rojo oscuro - < 10%

                    # Actualizar texto de porcentaje en la barra
                    artist['text_barra_pct'].set_text(f'{pct_contactos:.1f}%')

                    updated_artists.extend([
                        artist['line'],
                        artist['text_llamadas'],
                        artist['text_contactos'],
                        artist['text_tasa'],
                        artist['text_progreso'],
                        artist['rect_progress'],
                        artist['text_barra_pct']
                    ])

            return updated_artists

        # Crear animación
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=self.total_frames,
            interval=1000/self.fps,
            blit=True,
            repeat=True
        )

        # Guardar si se especifica
        if guardar_como:
            print(f"  💾 Guardando animación como {guardar_como}...")
            if guardar_como.endswith('.gif'):
                anim.save(guardar_como, writer='pillow', fps=self.fps)
            elif guardar_como.endswith('.mp4'):
                anim.save(guardar_como, writer='ffmpeg', fps=self.fps)
            print(f"  ✓ Animación guardada")

        plt.tight_layout()
        print("✓ Animación creada")

        return anim


class StaticVisualizer:
    """
    Clase para crear visualizaciones estáticas comparativas.
    """

    @staticmethod
    def plot_curvas_comparativas(resultados_simulacion, guardar=False):
        """
        Crea gráficos estáticos de comparación.

        Args:
            resultados_simulacion (dict): Resultados de simulación
            guardar (bool): Si guardar como imagen
        """
        print("\n📊 Generando gráficos comparativos...")

        estrategias = list(resultados_simulacion.keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Curvas de progreso
        ax = axes[0, 0]
        
        # Definir colores y estilos específicos por tipo de estrategia
        color_map = {
            'Aleatorio': '#E74C3C',  # Rojo - peor caso
            'Criterio_Empresarial': '#F39C12',  # Naranja - regla de negocio
            'Balanceado': '#3498DB',  # Azul - modelo ML
            'Alto_Recall': '#2ECC71',  # Verde - modelo ML
            'Alta_Precision': '#9B59B6'  # Morado - modelo ML
        }
        
        for nombre, resultado in resultados_simulacion.items():
            df_sim = resultado['df_simulacion']
            
            # Asignar color específico o por defecto
            if nombre in color_map:
                color = color_map[nombre]
            else:
                color = None
            
            # Estilo de línea
            if nombre == 'Aleatorio':
                linestyle = '--'
                linewidth = 2
                alpha = 0.6
            elif nombre == 'Criterio_Empresarial':
                linestyle = '-.'
                linewidth = 2.5
                alpha = 0.8
            else:  # Modelos ML
                linestyle = '-'
                linewidth = 3
                alpha = 0.9

            ax.plot(df_sim['porcentaje_llamadas'], df_sim['porcentaje_contactos'],
                   label=nombre, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=alpha)

        ax.set_xlabel('% del Dataset Recorrido', fontsize=12)
        ax.set_ylabel('% de Contactos Encontrados', fontsize=12)
        ax.set_title('Curvas de Búsqueda de Contactos', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])

        # Líneas de referencia
        for hito in [50, 80, 95, 99]:
            ax.axhline(y=hito, color='gray', linestyle=':', alpha=0.2)
            ax.axvline(x=hito, color='gray', linestyle=':', alpha=0.2)

        # 2. Llamadas necesarias por hito
        ax = axes[0, 1]
        hitos = [50, 80, 95, 99, 100]
        x = np.arange(len(hitos))
        width = 0.8 / len(estrategias)
        
        # Mismos colores que las curvas
        color_map_bars = {
            'Aleatorio': '#E74C3C',
            'Criterio_Empresarial': '#F39C12',
            'Balanceado': '#3498DB',
            'Alto_Recall': '#2ECC71',
            'Alta_Precision': '#9B59B6'
        }

        for i, nombre in enumerate(estrategias):
            resultado = resultados_simulacion[nombre]
            llamadas = [resultado['hitos'][h]['llamadas'] for h in hitos]
            
            color = color_map_bars.get(nombre, None)
            ax.bar(x + i*width, llamadas, width, label=nombre, alpha=0.8, color=color)

        ax.set_xlabel('Objetivo de Contactos Encontrados', fontsize=12)
        ax.set_ylabel('Llamadas Necesarias', fontsize=12)
        ax.set_title('Llamadas Necesarias por Objetivo', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(estrategias)-1) / 2)
        ax.set_xticklabels([f'{h}%' for h in hitos])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # 3. Tasa de éxito en primeros X%
        ax = axes[1, 0]
        categorias = ['Top 10%', 'Top 20%', 'Top 50%']
        x = np.arange(len(categorias))

        for i, nombre in enumerate(estrategias):
            resultado = resultados_simulacion[nombre]
            tasas = [
                resultado['tasa_top_10'],
                resultado['tasa_top_20'],
                resultado['tasa_top_50']
            ]

            color = color_map_bars.get(nombre, None)
            ax.bar(x + i*width, tasas, width, label=nombre, alpha=0.8, color=color)

        ax.set_xlabel('Segmento del Dataset', fontsize=12)
        ax.set_ylabel('Tasa de Éxito (%)', fontsize=12)
        ax.set_title('Tasa de Éxito por Segmento', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(estrategias)-1) / 2)
        ax.set_xticklabels(categorias)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # 4. Comparación de eficiencia (llamadas para 80%, 99% y 100%)
        ax = axes[1, 1]
        objetivos = ['80% contactos', '99% contactos', '100% contactos']
        x = np.arange(len(objetivos))

        for i, nombre in enumerate(estrategias):
            resultado = resultados_simulacion[nombre]
            llamadas = [
                resultado['hitos'][80]['llamadas'],
                resultado['hitos'][99]['llamadas'],
                resultado['hitos'][100]['llamadas']
            ]

            color = color_map_bars.get(nombre, None)
            ax.bar(x + i*width, llamadas, width, label=nombre, alpha=0.8, color=color)

        ax.set_xlabel('Objetivo', fontsize=12)
        ax.set_ylabel('Llamadas Necesarias', fontsize=12)
        ax.set_title('Eficiencia: Llamadas por Objetivo', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(estrategias)-1) / 2)
        ax.set_xticklabels(objetivos)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if guardar:
            plt.savefig('comparacion_estrategias.png', dpi=300, bbox_inches='tight')
            print("✓ Gráfico guardado como 'comparacion_estrategias.png'")

        plt.show()
        print("✓ Gráficos generados")

    @staticmethod
    def plot_mejora_vs_aleatorio(resultados_simulacion):
        """
        Visualiza la mejora de cada modelo respecto al aleatorio.

        Args:
            resultados_simulacion (dict): Resultados de simulación
        """
        if 'Aleatorio' not in resultados_simulacion:
            print("⚠️  No hay resultados aleatorios para comparar")
            return

        print("\n📈 Generando análisis de mejora vs Aleatorio...")

        aleatorio = resultados_simulacion['Aleatorio']
        estrategias = [k for k in resultados_simulacion.keys() if k != 'Aleatorio']

        # Calcular mejoras
        datos_mejora = []
        for nombre in estrategias:
            resultado = resultados_simulacion[nombre]

            for hito in [50, 80, 95, 99, 100]:
                llamadas_modelo = resultado['hitos'][hito]['llamadas']
                llamadas_aleatorio = aleatorio['hitos'][hito]['llamadas']
                reduccion = llamadas_aleatorio - llamadas_modelo
                pct_mejora = (reduccion / llamadas_aleatorio) * 100

                datos_mejora.append({
                    'Estrategia': nombre,
                    'Objetivo': f'{hito}%',
                    'Reduccion_llamadas': reduccion,
                    'Mejora_%': pct_mejora
                })

        df_mejora = pd.DataFrame(datos_mejora)

        # Color map consistente con otros gráficos
        color_map_mejora = {
            'Criterio_Empresarial': '#F39C12',
            'Balanceado': '#3498DB',
            'Alto_Recall': '#2ECC71',
            'Alta_Precision': '#9B59B6'
        }

        # Visualizar
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico 1: Reducción absoluta
        pivot_abs = df_mejora.pivot(index='Objetivo', columns='Estrategia', values='Reduccion_llamadas')
        colors = [color_map_mejora.get(col, '#95A5A6') for col in pivot_abs.columns]  # Gris por defecto
        pivot_abs.plot(kind='bar', ax=axes[0], width=0.8, color=colors)
        axes[0].set_xlabel('Objetivo de Contactos', fontsize=12)
        axes[0].set_ylabel('Reducción de Llamadas vs Aleatorio', fontsize=12)
        axes[0].set_title('Mejora Absoluta vs Aleatorio', fontsize=14, fontweight='bold')
        axes[0].legend(title='Estrategia', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)

        # Gráfico 2: Mejora porcentual
        pivot_pct = df_mejora.pivot(index='Objetivo', columns='Estrategia', values='Mejora_%')
        pivot_pct.plot(kind='bar', ax=axes[1], width=0.8, color=colors)
        axes[1].set_xlabel('Objetivo de Contactos', fontsize=12)
        axes[1].set_ylabel('Mejora (%) vs Aleatorio', fontsize=12)
        axes[1].set_title('Mejora Porcentual vs Aleatorio', fontsize=14, fontweight='bold')
        axes[1].legend(title='Estrategia', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.show()

        print("✓ Análisis de mejora generado")

        return df_mejora
