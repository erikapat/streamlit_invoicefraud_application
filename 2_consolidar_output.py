# -*- coding: utf-8 -*-
import os
import pandas as pd

# Ruta raíz donde están las carpetas Batch_real, Etiquetadas, Sinteticas
base_dir = r"C:\Users\epgonz1\MAPFRE\DCD - Externos - KEEPLER - KEEPLER\CAPTURA TEXTO Y FRAUDE\EVOLUCIÓN\Resultados_ejecuciones\Entregas\25062025\\"
#base_dir = r'C:\Users\epgonz1\MAPFRE\DCD - Externos - KEEPLER - KEEPLER\CAPTURA TEXTO Y FRAUDE\EVOLUCIÓN\Resultados_ejecuciones\\'
#base_dir = 'C:/Users/epgonz1/MAPFRE/DCD - Externos - KEEPLER - KEEPLER/CAPTURA TEXTO Y FRAUDE/Validación de facturas - Team Mapfre/Hito 3/Resultado_hito3_26_05_24/'

# Nombres de las carpetas a procesar
tipos = ['Batch_real', 'Etiquetadas', 'Sinteticas']

# Diccionario para guardar los DataFrames concatenados
df_concat = {}

# Recorremos cada tipo
for tipo in tipos:
    ruta_tipo = os.path.join(base_dir, tipo)
    csvs = []

    for root, _, files in os.walk(ruta_tipo):
        for file in files:
            if file.endswith('.csv'):
                ruta_csv = os.path.join(root, file)
                print(ruta_csv)
                try:
                    df = pd.read_csv(ruta_csv)

                    # Extraer año, mes, día desde el path
                    partes = os.path.normpath(ruta_csv).split(os.sep)
                    if 'results' in partes:
                        i = partes.index('results')
                        anio, mes, dia = partes[i + 1], partes[i + 2], partes[i + 3]
                        # Crear columna única de fecha con formato YYYY-MM-DD
                        df['fecha'] = f"{anio}-{mes.zfill(2)}-{dia.zfill(2)}"
                        anio, mes, dia = partes[i+1], partes[i+2], partes[i+3]
                        df['año'] = anio
                        df['mes'] = mes
                        df['dia'] = dia
                    else:
                        print(f"Ruta no contiene 'results': {ruta_csv}")
                        df['fecha'] = None

                    csvs.append(df)
                except Exception as e:
                    print(f"Error leyendo {ruta_csv}: {e}")

    if csvs:
        df_concat[tipo] = pd.concat(csvs, ignore_index=True)
        # Guardamos el CSV final
        output_path = os.path.join(base_dir, f"{tipo.lower()}_final.csv")
        df_concat[tipo].to_csv(output_path, index=False)
        print(f"Guardado: {output_path}")
    else:
        print(f"No se encontraron CSVs para {tipo}")

print("Proceso completado.")
