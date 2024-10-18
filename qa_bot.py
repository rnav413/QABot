import streamlit as st
import pandas as pd
import io
import json
import math
import numpy as np
from groq import Groq

# Inicializar la API de Groq
groq = Groq(
    api_key="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",
)

def gherkin_agent(steps):
    steps_to_send = steps[1:]

    def sanitize_step(step):
        for key, value in step.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                step[key] = ""
        return step

    sanitized_steps = [sanitize_step(step) for step in steps_to_send]
    steps_json = json.dumps(sanitized_steps)

    system_prompt = """
    You are a Gherkin expert agent. You will receive a JSON containing the steps of a test case in the following format:

    [
      {
        'Test Step': <Step Number>,
        'Step Action': '<Step Action>',
        'Step Expected': '<Expected Result>',
      },
      ...
    ]

    Your task is to analyze each step and correct it according to the Gherkin format, returning the steps in the same JSON format but adapted. Make sure to use the appropriate Gherkin keywords such as Given, When, Then, And, But. Do not include any additional information outside of the requested format.
    Please provide only the corrected steps in JSON format.
    """

    try:
        question = groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": steps_json},
            ],
            model="llama3-70b-8192",
            temperature=1.2,
            max_tokens=8192,
            top_p=1,
            stop=None,
            stream=False
        )

        response_content = question.choices[0].message.content
        processed_steps = json.loads(response_content)

    except json.JSONDecodeError as e:
        processed_steps = []
    except Exception as e:
        processed_steps = []

    return [steps[0]] + processed_steps

def process_test_cases(df):
    test_cases = []
    current_test_case = None

    for index, row in df.iterrows():
        if pd.notna(row['ID']):
            if current_test_case is not None:
                test_cases.append(current_test_case)
            current_test_case = {
                'ID': row['ID'],
                'Work Item Type': row['Work Item Type'],
                'Title': row['Title'],
                'Priority': row['Priority'],
                'Steps': [],
                'Area Path': row['Area Path'],
                'Assigned To': row['Assigned To'],
                'State': row['State'],
                'Tags': row['Tags'],
            }
        else:
            if current_test_case is not None:
                step = {
                    'Test Step': row['Test Step'],
                    'Step Action': row['Step Action'],
                    'Step Expected': row['Step Expected'],
                }
                current_test_case['Steps'].append(step)

    if current_test_case is not None:
        test_cases.append(current_test_case)

    for tc in test_cases:
        tc["Steps"] = gherkin_agent(tc["Steps"])

    rows = []

    for tc in test_cases:
        tc_row = {
            'ID': tc.get('ID', np.nan),
            'Work Item Type': tc.get('Work Item Type', np.nan),
            'Title': tc.get('Title', np.nan),
            'Test Step': np.nan,
            'Step Action': np.nan,
            'Step Expected': np.nan,
            'Priority': tc.get('Priority', np.nan),
            'Area Path': tc.get('Area Path', np.nan),
            'Assigned To': tc.get('Assigned To', np.nan),
            'State': tc.get('State', np.nan),
            'Tags': tc.get('Tags', np.nan),
        }
        rows.append(tc_row)

        for step in tc.get('Steps', []):
            step_row = {
                'ID': np.nan,
                'Work Item Type': np.nan,
                'Title': np.nan,
                'Test Step': step.get('Test Step', np.nan),
                'Step Action': step.get('Step Action', np.nan),
                'Step Expected': step.get('Step Expected', np.nan),
                'Priority': np.nan,
                'Area Path': np.nan,
                'Assigned To': np.nan,
                'State': np.nan,
                'Tags': np.nan,
            }
            rows.append(step_row)

    processed_df = pd.DataFrame(rows)
    return processed_df

def main():
    st.title("Formato de Casos de Prueba Gherkin")
    st.write("Sube un archivo Excel o CSV que contenga tus casos de prueba, y recibe una versión corregida con los pasos formateados según la sintaxis Gherkin.")

    uploaded_file = st.file_uploader("Elige un archivo Excel o CSV", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        st.write("**Archivo cargado exitosamente.**")

        if st.button("Procesar Archivo"):
            with st.spinner('Procesando...'):
                try:
                    # Leer el archivo cargado en un DataFrame
                    if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        st.error('Formato de archivo no válido. Solo se admiten .xlsx, .xls y .csv.')
                        return

                    # Procesar el DataFrame
                    processed_df = process_test_cases(df)

                    st.success('¡Procesamiento completado!')

                    # Convertir el DataFrame procesado a un archivo Excel en memoria
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        processed_df.to_excel(writer, index=False)
                    output.seek(0)

                    # Mostrar el DataFrame procesado (opcional)
                    st.write("**Datos Procesados:**")
                    st.dataframe(processed_df)

                    # Botón para descargar el archivo
                    st.download_button(
                        label="Descargar Archivo Excel Procesado",
                        data=output.getvalue(),
                        file_name="casos_de_prueba_procesados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    st.error(f'Ocurrió un error: {e}')

if __name__ == "__main__":
    main()
