import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud

# Load models
diabetes_model = pickle.load(open("/home/pavani/Downloads/diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("/home/pavani/Downloads/heart_disease_model.sav", 'rb'))
parkinsons_model = pickle.load(open("/home/pavani/Downloads/parkinsons_model.sav", 'rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu('Multiple disease prediction',
                           ['Home', 'Diabetes prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Data Visualiser'],
                           icons=['house', 'activity', 'heart', 'person',
                                  'person-arms-up', 'bar-chart-line-fill'],
                           default_index=0)

# Initialize session state for input values
if 'input_values' not in st.session_state:
    st.session_state.input_values = {}

# Home Page
if selected == 'Home':
    
    st.title("Disease Prediction System")
    st.markdown("## Welcome to the predictive modelling...!")
    st.write("Our system helps disease predicting using advanced machine learning models. Explore the app to know more!")
    st.markdown("""
    <div style='text-align: center;'>
        <h3>Explore Our Features:</h3>
        <ul style='list-style-type: none; padding: 0;'>
            <li style='margin: 10px 0;'><button id='traffic-button' style='padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Disease Prediction</button></li>
            <li style='margin: 10px 0;'><button id='data-button' style='padding: 10px 20px; font-size: 16px; background-color: #008CBA; color: white; border: none; border-radius: 5px;'>Visualizer</button></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <script>
    document.getElementById('traffic-button').onclick = function() {
        alert('Redirecting to Lane detection...');
    };
    document.getElementById('data-button').onclick = function() {
        alert('Redirecting to Object detection...');
    };
    </script>
    """, unsafe_allow_html=True)

# Function to convert inputs to floats with checks
def convert_to_float(input_list):
    try:
        return [float(x) for x in input_list if x.strip() != '']
    except ValueError:
        st.error("Please fill all the fields with valid numbers.")
        return None

# Diabetes Prediction Page
if selected == 'Diabetes prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', st.session_state.input_values.get('Pregnancies', ''))
    with col2:
        Glucose = st.text_input('Glucose Level', st.session_state.input_values.get('Glucose', ''))
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', st.session_state.input_values.get('BloodPressure', ''))
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', st.session_state.input_values.get('SkinThickness', ''))
    with col2:
        Insulin = st.text_input('Insulin Level', st.session_state.input_values.get('Insulin', ''))
    with col3:
        BMI = st.text_input('BMI value', st.session_state.input_values.get('BMI', ''))
    with col1:
        DPF = st.text_input('Diabetes Pedigree Function', st.session_state.input_values.get('DPF', ''))
    with col2:
        Age = st.text_input('Age', st.session_state.input_values.get('Age', ''))

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]

        st.session_state.input_values['Pregnancies'] = Pregnancies
        st.session_state.input_values['Glucose'] = Glucose
        st.session_state.input_values['BloodPressure'] = BloodPressure
        st.session_state.input_values['SkinThickness'] = SkinThickness
        st.session_state.input_values['Insulin'] = Insulin
        st.session_state.input_values['BMI'] = BMI
        st.session_state.input_values['DPF'] = DPF
        st.session_state.input_values['Age'] = Age

        user_input = convert_to_float(user_input)
        if user_input is not None:
            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is Diabetic'
            else:
                diab_diagnosis = 'The person is not Diabetic'
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)
    with col1:
        age1 = st.text_input('Age', st.session_state.input_values.get('age1', ''))
    with col2:
        sex1 = st.text_input('Sex', st.session_state.input_values.get('sex1', ''))
    with col3:
        cp = st.text_input('Chest pain types', st.session_state.input_values.get('cp', ''))
    with col1:
        trestbps = st.text_input('Resting Blood pressure', st.session_state.input_values.get('trestbps', ''))
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', st.session_state.input_values.get('chol', ''))
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120mg/dl', st.session_state.input_values.get('fbs', ''))
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results', st.session_state.input_values.get('restecg', ''))
    with col2:
        thalach = st.text_input('Maximum HeartRate achieved', st.session_state.input_values.get('thalach', ''))
    with col3:
        exang = st.text_input('Exercise induced Angina', st.session_state.input_values.get('exang', ''))
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise', st.session_state.input_values.get('oldpeak', ''))
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment', st.session_state.input_values.get('slope', ''))
    with col3:
        ca = st.text_input('Major vessel colored by fluoroscopy', st.session_state.input_values.get('ca', ''))
    with col1:
        thal = st.text_input('thal: 0=normal,1=fixed defect,2=reversible defect', st.session_state.input_values.get('thal', ''))

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age1, sex1, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = convert_to_float(user_input)
        if user_input is not None:
            st.session_state.input_values['age1'] = age1
            st.session_state.input_values['sex1'] = sex1
            st.session_state.input_values['cp'] = cp
            st.session_state.input_values['trestbps'] = trestbps
            st.session_state.input_values['chol'] = chol
            st.session_state.input_values['fbs'] = fbs
            st.session_state.input_values['restecg'] = restecg
            st.session_state.input_values['thalach'] = thalach
            st.session_state.input_values['exang'] = exang
            st.session_state.input_values['oldpeak'] = oldpeak
            st.session_state.input_values['slope'] = slope
            st.session_state.input_values['ca'] = ca
            st.session_state.input_values['thal'] = thal

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person does not have heart disease'
    st.success(heart_diagnosis)

# Parkinsons Prediction Page
if selected == 'Parkinsons Prediction':
    st.title('Parkinsons Prediction')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', st.session_state.input_values.get('fo', ''))
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', st.session_state.input_values.get('fhi', ''))
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', st.session_state.input_values.get('flo', ''))
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)', st.session_state.input_values.get('Jitter_percent', ''))
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', st.session_state.input_values.get('Jitter_Abs', ''))
    with col1:
        RAP = st.text_input('MDVP:RAP', st.session_state.input_values.get('RAP', ''))
    with col2:
        PPQ = st.text_input('MDVP:PPQ', st.session_state.input_values.get('PPQ', ''))
    with col3:
        DDP = st.text_input('Jitter:DDP', st.session_state.input_values.get('DDP', ''))
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer', st.session_state.input_values.get('Shimmer', ''))
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', st.session_state.input_values.get('Shimmer_dB', ''))
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3', st.session_state.input_values.get('APQ3', ''))
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5', st.session_state.input_values.get('APQ5', ''))
    with col3:
        APQ = st.text_input('MDVP:APQ', st.session_state.input_values.get('APQ', ''))
    with col4:
        DDA = st.text_input('Shimmer:DDA', st.session_state.input_values.get('DDA', ''))
    with col5:
        NHR = st.text_input('NHR', st.session_state.input_values.get('NHR', ''))
    with col1:
        HNR = st.text_input('HNR', st.session_state.input_values.get('HNR', ''))
    with col2:
        RPDE = st.text_input('RPDE', st.session_state.input_values.get('RPDE', ''))
    with col3:
        DFA = st.text_input('DFA', st.session_state.input_values.get('DFA', ''))
    with col4:
        spread1 = st.text_input('spread1', st.session_state.input_values.get('spread1', ''))
    with col5:
        spread2 = st.text_input('spread2', st.session_state.input_values.get('spread2', ''))
    with col1:
        D2 = st.text_input('D2', st.session_state.input_values.get('D2', ''))
    with col2:
        PPE = st.text_input('PPE', st.session_state.input_values.get('PPE', ''))

    parkinsons_diagnosis = ''
    if st.button('Parkinsons Test Result'):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = convert_to_float(user_input)
        if user_input is not None:
            st.session_state.input_values['fo'] = fo
            st.session_state.input_values['fhi'] = fhi
            st.session_state.input_values['flo'] = flo
            st.session_state.input_values['Jitter_percent'] = Jitter_percent
            st.session_state.input_values['Jitter_Abs'] = Jitter_Abs
            st.session_state.input_values['RAP'] = RAP
            st.session_state.input_values['PPQ'] = PPQ
            st.session_state.input_values['DDP'] = DDP
            st.session_state.input_values['Shimmer'] = Shimmer
            st.session_state.input_values['Shimmer_dB'] = Shimmer_dB
            st.session_state.input_values['APQ3'] = APQ3
            st.session_state.input_values['APQ5'] = APQ5
            st.session_state.input_values['APQ'] = APQ
            st.session_state.input_values['DDA'] = DDA
            st.session_state.input_values['NHR'] = NHR
            st.session_state.input_values['HNR'] = HNR
            st.session_state.input_values['RPDE'] = RPDE
            st.session_state.input_values['DFA'] = DFA
            st.session_state.input_values['spread1'] = spread1
            st.session_state.input_values['spread2'] = spread2
            st.session_state.input_values['D2'] = D2
            st.session_state.input_values['PPE'] = PPE

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'The person has Parkinsons disease'
            else:
                parkinsons_diagnosis = 'The person does not have Parkinsons disease'
    st.success(parkinsons_diagnosis)

if selected == 'Data Visualiser':
    # st.set_page_config(page_title='Data Visualizer', layout='centered', page_icon='📊')
    st.title('📊 Data Visualizer')
    
    working_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = f'{working_dir}/Data'
    files_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    selected_file = st.selectbox('Select a file', files_list, index=None)
    
    if selected_file:
        file_path = os.path.join(folder_path, selected_file)
        df = pd.read_csv(file_path)

        col1, col2 = st.columns(2)
        columns = df.columns.tolist()

        with col1:
            st.write(df.head())
        with col2:
            x_axis = st.selectbox('Select the x-axis', options=columns + ['None'], index=None)
            y_axis = st.selectbox('Select the Y-axis', options=columns + ['None'], index=None)

        plot_list = ['Bar Chart', 'Scatter Plot', 'Count Plot']
        selected_plot = st.selectbox('Select a Plot', options=plot_list, index=None)

        if st.button('Generate Plot'):
            fig, ax = plt.subplots(figsize=(6, 4))

            if selected_plot == 'Bar Chart':
                sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
            elif selected_plot == 'Scatter Plot':
                sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
            elif selected_plot == 'Count Plot':
                sns.countplot(x=df[x_axis], ax=ax)

            st.pyplot(fig)

