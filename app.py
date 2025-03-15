import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load artifacts
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Feature configuration (EXACT MATCH WITH TRAINING DATA)
COLLECTED_FEATURES = {
    'Subjects_before_university': [
        'Chemistry', 'Physics', 'Mathematics', 'Biology',
        'Computer Science', 'Engineering', 'Psychology',
        'Sociology', 'Geography', 'Geology', 
        'Environmental Science', 'Computer Engineering',
        'Information Technology', 'Software Engineering',
        'Unknown', 'Other'
    ],
    'Career_prospects_Importance': [
        'Extremely not important', 'Somewhat not important',
        'Neutral', 'Somewhat important', 'Extremely important'
    ],
    'Post_course_goals': [
        'Secure a job in a specific industry or field', 
        'Develop transferable skills for various job sectors',
        'Pursue further study or research in the field',
        'Start my own business or entrepreneurial venture',
        'Gain a broad knowledge base and personal growth'
    ],
    'Workload_assessment_method': [
        'I didn’t think about it much; I just went for it',
        'I assumed it would be manageable based on my interests and strengths',
        'I considered the course prerequisites and my academic background',
        'I researched online and talked to current students',
        'I spoke with academic advisors or professors'
    ],
    'Preferred_learning_environment': [
        'A hands-on, practical approach with real-world applications',
        'A theoretical, lecture-based approach with in-depth learning',
        'A mix of both theory and practice',
        'Self-paced, independent learning',
        'A collaborative, group-based learning experience'
    ],
    'Alignment_with_career_goals': [
        'Neutral', 
        'Somewhat interested',
        'Extremely interested',
        'Extremely not interested',
        'Somewhat not interested'
    ],
    'Confident_skills': [
        'Problem-solving',
        'Researching skills',
        'Mathematical skills',
        'technical skills'
    ],
    'Your_strengths': [
        'Data analysis',
        'Lab work',
        'Ethical reasoning',
        'Leadership skills'
    ]
}

# Default values from training
DEFAULTS = {
    'Reason_for_current_course': 'Passion for the subject',
    'First_interest_for_current_course': 'Personal experience or hobby',
    'previous_course_prepared_how_much_for_current_course': 5.71,
    'Would_you_recommend': 'Yes',
    'Alternative_choice': 'No',
    'satisfaction_with_current_course': 5.65  # Keep but handle properly
}

def sanitize(name):
    """EXACT same sanitization as training"""
    return (name.replace(" ", "_")
              .replace(",", "")
              .replace(";", "")
              .replace("’", "")
              .replace("-", "_")
              .lower())

def preprocess_input(form_data):
    """Process form data to match model requirements"""
    df = pd.DataFrame([form_data])
    
    # Add defaults for missing features
    for feature, value in DEFAULTS.items():
        df[feature] = value
    
    # Ordinal encoding with training order
    ordinal_map = {
        'Career_prospects_Importance': COLLECTED_FEATURES['Career_prospects_Importance'],
        'Alignment_with_career_goals': COLLECTED_FEATURES['Alignment_with_career_goals'],
        'Workload_assessment_method': COLLECTED_FEATURES['Workload_assessment_method']
    }
    for col, categories in ordinal_map.items():
        df[col] = categories.index(form_data[col])
    
    # One-hot encoding with exact training names
    categorical_cols = [
        'Subjects_before_university', 'Post_course_goals',
        'Preferred_learning_environment', 'Confident_skills',
        'Your_strengths', 'Reason_for_current_course',
        'First_interest_for_current_course', 'Would_you_recommend',
        'Alternative_choice'
    ]
    
    for col in categorical_cols:
        options = COLLECTED_FEATURES.get(col, [DEFAULTS.get(col, '')])
        for option in options:
            clean_option = sanitize(option)
            df[f"{col}_{clean_option}"] = 0
        selected = form_data.get(col, DEFAULTS.get(col, ''))
        clean_selected = sanitize(selected)
        df[f"{col}_{clean_selected}"] = 1
    
    # Scale ONLY features scaled during training
    numerical_cols = ['previous_course_prepared_how_much_for_current_course']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Add satisfaction WITHOUT scaling
    df['satisfaction_with_current_course'] = DEFAULTS['satisfaction_with_current_course']
    
    # Match EXACT feature names
    expected_features = model.feature_names_in_
    missing = set(expected_features) - set(df.columns)
    for feature in missing:
        df[feature] = 0
    
    return df[expected_features]

def main():
    st.title("University Course Prediction System")
    
    with st.form("student_form"):
        st.header("Prospective Student Information")
        
        form_data = {
            'Subjects_before_university': st.selectbox(
                "Main subjects studied before university:",
                COLLECTED_FEATURES['Subjects_before_university']
            ),
            'Career_prospects_Importance': st.selectbox(
                "Career prospects importance:",
                COLLECTED_FEATURES['Career_prospects_Importance']
            ),
            'Post_course_goals': st.selectbox(
                "Post-course goals:",
                COLLECTED_FEATURES['Post_course_goals']
            ),
            'Workload_assessment_method': st.selectbox(
                "Workload assessment method:",
                COLLECTED_FEATURES['Workload_assessment_method']
            ),
            'Preferred_learning_environment': st.selectbox(
                "Preferred learning environment:",
                COLLECTED_FEATURES['Preferred_learning_environment']
            ),
            'Alignment_with_career_goals': st.selectbox(
                "Career goals alignment:",
                COLLECTED_FEATURES['Alignment_with_career_goals']
            ),
            'Confident_skills': st.selectbox(
                "Most confident skills:",
                COLLECTED_FEATURES['Confident_skills']
            ),
            'Your_strengths': st.selectbox(
                "Your strengths:",
                COLLECTED_FEATURES['Your_strengths']
            )
        }
        
        submitted = st.form_submit_button("Predict Course")

    if submitted:
        with st.spinner('Analyzing your profile...'):
            try:
                processed_data = preprocess_input(form_data)
                
                # Debug: Show first 5 features
                st.write("### Processed Features Preview")
                st.dataframe(processed_data.iloc[:, :5])
                
                prediction = model.predict(processed_data)
                proba = model.predict_proba(processed_data)
                course = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(proba) * 100
                
                st.success(f"## Recommended Course: {course}")
                st.metric("Confidence Score", f"{confidence:.1f}%")
                
                with st.expander("Detailed Analysis"):
                    st.write("**Key Factors:**")
                    st.write(f"- Career Focus: {form_data['Career_prospects_Importance']}")
                    st.write(f"- Learning Style: {form_data['Preferred_learning_environment']}")
                    st.write(f"- Strength Match: {form_data['Your_strengths']}")
                    
                    st.write("\n**Full Probability Distribution:**")
                    proba_df = pd.DataFrame({
                        'Course': label_encoder.classes_,
                        'Probability (%)': (proba[0] * 100).round(1)
                    }).sort_values('Probability (%)', ascending=False)
                    st.dataframe(proba_df)
                
                st.balloons()

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()