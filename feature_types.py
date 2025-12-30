def get_feature_types():
    quantitative = [
        'age',
        'avg_glucose_level',
        'bmi'
    ]

    categorical = [
        'gender',
        'hypertension',
        'heart_disease',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status'
    ]

    print("Quantitative Features:")
    for q in quantitative:
        print("-", q)

    print("\nCategorical Features:")
    for c in categorical:
        print("-", c)

    return quantitative, categorical
