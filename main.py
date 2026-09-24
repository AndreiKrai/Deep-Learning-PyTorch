from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset

# 2. Підготовка даних:
# Завантажте набір даних Concrete Strength Prediction з платформи Kaggle.
dataset_path = Path(
    kagglehub.dataset_download("mchilamwar/predict-concrete-strength")
)

print("Path to dataset files:", dataset_path)

csv_file = next(dataset_path.glob("*.csv"))
df = pd.read_csv(csv_file)

# print(df.sample(5, random_state=42))
# df.info()

# Розділіть дані на ознаки (X) та цільову змінну (y).
X = df.drop(columns=["Strength"])
y = df["Strength"]

# Розділіть дані на навчальний та тестовий набори.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)
# Нормалізуйте вхідні дані за допомогою StandardScaler.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Створення моделі:
#  Підготовка моделі
class LinearModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=20, second_hidden_dim=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Linear(second_hidden_dim, 1),
        )

    def forward(self, x):
        return self.features(x)


model = LinearModel(in_dim=X_train.shape[1])
print(model)

# 4. Налаштування навчання:
# Виберіть функцію втрат (наприклад, MSELoss для регресії). Обґрунтуйте вибір функції втрат.
criterion = nn.MSELoss()  # MSELoss підходить для задач регресії, оскільки вона вимірює середньоквадратичну помилку між передбаченими та фактичними значеннями.

# Встановіть гіперпараметри — швидкість навчання (lr), розмір батчу, кількість епох.
lr = 0.001
batch_size = 32
num_epochs = 100

# Виберіть оптимізатор (наприклад, Adam) та налаштуйте його параметри.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam є популярним оптимізатором для глибокого навчання, оскільки він адаптивно налаштовує швидкість навчання для кожного параметра.

# 5. Навчання моделі:
# Перетворіть дані на тензори PyTorch.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Навчання моделі.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    average_loss = epoch_loss / len(train_loader.dataset)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {average_loss:.4f}"
        )

# 6. Оцінка моделі:
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)

# Перетворюємо тензори на одновимірні масиви NumPy для scikit-learn.
predictions_np = predictions.numpy().ravel()
y_test_np = y_test_tensor.numpy().ravel()

# MSE сильніше штрафує великі помилки.
mse = mean_squared_error(y_test_np, predictions_np)
# MAE показує середню абсолютну помилку в одиницях Strength.
mae = mean_absolute_error(y_test_np, predictions_np)
# R² показує, яку частину варіації Strength пояснює модель.
r2 = r2_score(y_test_np, predictions_np)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")