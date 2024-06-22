import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Broj redova u train skupu: {len(train_df)}")
    print(f"Broj redova u test skupu: {len(test_df)}")

    X_train = train_df.drop('Stage', axis=1)
    y_train = train_df['Stage']
    X_test = test_df.drop('Stage', axis=1)
    y_test = test_df['Stage']

    categorical_features = ['Gender', 'Smoking_History', 'Tumor_Location', 'Treatment', 'Ethnicity', 'Insurance_Type', 'Family_History']
    numerical_features = ['Age', 'Tumor_Size_mm', 'Survival_Months', 'Performance_Status', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
                          'Blood_Pressure_Pulse', 'Hemoglobin_Level', 'White_Blood_Cell_Count', 'Platelet_Count', 'Albumin_Level', 
                          'Alkaline_Phosphatase_Level', 'Alanine_Aminotransferase_Level', 'Aspartate_Aminotransferase_Level', 
                          'Creatinine_Level', 'LDH_Level', 'Calcium_Level', 'Phosphorus_Level', 'Glucose_Level', 'Potassium_Level', 
                          'Sodium_Level', 'Smoking_Pack_Years']

    # Enkodiranje kategorijskih obeležja
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cats = encoder.fit_transform(train_df[categorical_features])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))

    # Spajanje numeričkih i enkodiranih kategorijskih obeležja
    combined_df = pd.concat([train_df[numerical_features].reset_index(drop=True), encoded_cats_df.reset_index(drop=True)], axis=1)

    # Analiza korelacije
    cor = combined_df.corr()
    upper_triangle = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

    # Uklanjanje visoko koreliranih obeležja
    reduced_df = combined_df.drop(columns=to_drop)

    # Kreiranje korrelacione matrice za kombinovana obeležja
    plt.figure(figsize=(20, 15))
    reduced_cor = reduced_df.corr()
    sns.heatmap(reduced_cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
    plt.show()

    # Ažuriranje liste numeričkih i kategorijskih obeležja nakon uklanjanja visoko koreliranih
    updated_numerical_features = [feature for feature in numerical_features if feature in reduced_df.columns]
    updated_categorical_features = [feature for feature in categorical_features if any([col for col in reduced_df.columns if col.startswith(feature)])]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())]), updated_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), updated_categorical_features)
        ])

    # Ažurirani pipeline sa najboljim parametrima za RandomForestClassifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            max_depth=30, 
            min_samples_split=5, 
            min_samples_leaf=1, 
            random_state=42))
    ])

    # Treniranje modela
    pipeline.fit(X_train, y_train)

    # Analiza značaja obeležja
    feature_importances = pipeline.named_steps['classifier'].feature_importances_
    feature_names = updated_numerical_features + list(encoder.get_feature_names_out(updated_categorical_features))
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

    # Prikaz značaja obeležja
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Značaj obeležja')
    plt.show()

    # Predikcija i evaluacija
    y_pred = pipeline.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 Score: {score}')
