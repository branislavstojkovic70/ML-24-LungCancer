import pandas as pd
from sklearn.model_selection import train_test_split

# Učitavanje CSV fajla u DataFrame
file_path = 'lung_cancer_data.csv'  # Zamijenite sa putanjom do vašeg CSV fajla
data = pd.read_csv(file_path)

# Podela podataka u train i test skupove sa odnosom 70-30
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Provera broja redova u train i test skupovima
print(f"Broj redova u train skupu: {len(train_data)}")
print(f"Broj redova u test skupu: {len(test_data)}")

# Spremanje podeljenih skupova u nove CSV fajlove (ako je potrebno)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
