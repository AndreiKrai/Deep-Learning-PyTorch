from pathlib import Path
import random
import shutil

import kagglehub
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


DATASET_HANDLE = "puneet6060/intel-image-classification"
DATASET_DIRECTORY = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 32
IMAGE_SIZE = (150, 150)
NUM_EPOCHS = 10
CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
REPORT_DIRECTORY = Path(__file__).resolve().parent / "reports"

# Kaggle dataset files are stored in data/, while trained weights are stored in checkpoints/.
# Download the dataset from Kaggle
def download_dataset() -> Path:
	"""Download the Intel Image Classification dataset from Kaggle."""
	DATASET_DIRECTORY.mkdir(parents=True, exist_ok=True)
	dataset_path = Path(
		kagglehub.dataset_download(
			DATASET_HANDLE,
			output_dir=str(DATASET_DIRECTORY),
		)
	)

	expected_directories = ("seg_train", "seg_test")
	missing_directories = [
		directory
		for directory in expected_directories
		if not (dataset_path / directory).is_dir()
	]
	if missing_directories:
		raise FileNotFoundError(
			f"Downloaded dataset is missing: {', '.join(missing_directories)}"
		)

	print(f"Dataset downloaded to: {dataset_path}")
	return dataset_path

# The original dataset has train and test data, so 20% of train is reserved for validation.
def split_dataset(dataset_path: Path) -> None:
	"""Create train, validation, and test folders inside data/."""
	train_source = dataset_path / "seg_train" / "seg_train"
	test_source = dataset_path / "seg_test" / "seg_test"

	random_generator = random.Random(42)
	for class_directory in sorted(train_source.iterdir()):
		if not class_directory.is_dir():
			continue

		# Shuffle files within each class to create a representative train/validation split.
		files = sorted(file for file in class_directory.iterdir() if file.is_file())
		random_generator.shuffle(files)
		split_index = int(len(files) * 0.8)

		# ImageFolder needs one subdirectory per class in every data split.
		for split_name, split_files in (
			("train", files[:split_index]),
			("val", files[split_index:]),
		):
			destination = DATASET_DIRECTORY / split_name / class_directory.name
			destination.mkdir(parents=True, exist_ok=True)
			for file in split_files:
				shutil.copy2(file, destination / file.name)

	# The Kaggle test set is already separated, so copy it without splitting.
	for class_directory in sorted(test_source.iterdir()):
		if not class_directory.is_dir():
			continue

		destination = DATASET_DIRECTORY / "test" / class_directory.name
		destination.mkdir(parents=True, exist_ok=True)
		for file in sorted(class_directory.iterdir()):
			if file.is_file():
				shutil.copy2(file, destination / file.name)

	print(f"Split dataset saved to: {DATASET_DIRECTORY}")


def create_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
	"""Load train, validation, and test images in batches."""
	# Apply identical preprocessing to every split so the model sees a consistent format.
	transform = transforms.Compose([
	# Resize images to the specified IMAGE_SIZE before converting them to tensors.
		transforms.Resize(IMAGE_SIZE),
		transforms.ToTensor(),
	# Normalize images using the mean and standard deviation of the ImageNet dataset.
		transforms.Normalize(
			mean=(0.485, 0.456, 0.406),
			std=(0.229, 0.224, 0.225),
		),
	])

	# ImageFolder converts class directory names into integer labels.
	datasets_by_split = {
		split: datasets.ImageFolder(DATASET_DIRECTORY / split, transform=transform)
		for split in ("train", "val", "test")
	}
	# Shuffle only training data; stable validation/test order makes evaluation reproducible.
	loaders = {
		split: DataLoader(
			dataset,
			batch_size=BATCH_SIZE,
			shuffle=split == "train",
		)
		for split, dataset in datasets_by_split.items()
	}

	print("Classes:", datasets_by_split["train"].classes)
	for split, dataset in datasets_by_split.items():
		print(f"{split}: {len(dataset)} images")

	return loaders["train"], loaders["val"], loaders["test"]


# Створення моделі: Варіант A: Проста згорткова нейронна мережа
class SimpleCNN(nn.Module):
	"""A small convolutional network for six image classes."""

	def __init__(self, num_classes: int) -> None:
		super().__init__()

		self.features = nn.Sequential(
			# Learn simple visual patterns such as edges and color changes.
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			# Combine the simple patterns into more complex image features.
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			# Extract high-level features useful for classifying scenes.
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			# 150x150 becomes 18x18 after three 2x2 pooling layers.
			nn.Linear(128 * 18 * 18, 256),
			nn.ReLU(),
			# Produce one output score for each image class.
			nn.Linear(256, num_classes),
		)

	def forward(self, images):
		# Return raw class scores (logits); CrossEntropyLoss handles their comparison.
		features = self.features(images)
		return self.classifier(features)

# Створення моделі: Варіант B: Transfer Learning
def create_transfer_model(num_classes: int) -> nn.Module:
	"""Create a ResNet18 pretrained on ImageNet for this classification task."""
	# Load visual features learned from the large ImageNet dataset.
	model = resnet18(weights=ResNet18_Weights.DEFAULT)

	# Freeze the pretrained feature extractor and train only the new classifier.
	for parameter in model.parameters():
		parameter.requires_grad = False

	# ResNet18 originally predicts 1000 ImageNet classes; replace that head.
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	# The replacement layer is trainable, while the pretrained feature extractor stays frozen.
	return model

# Визначення функції втрат та оптимізатора:
def create_training_components(
	model: nn.Module,
) -> tuple[nn.Module, torch.optim.Optimizer]:
	"""Create the loss function and optimizer for a classification model."""
	# CrossEntropyLoss combines LogSoftmax and NLLLoss and expects class indices.
	# It is appropriate because this is a single-label, six-class problem.
# 	задача має 6 взаємовиключних класів;
# кожне зображення належить лише одному класу;
# функція працює безпосередньо з logits моделі та індексом правильного класу;
# Softmax вручну додавати не потрібно.
	criterion = nn.CrossEntropyLoss()

	# Optimize only trainable parameters; this skips frozen ResNet18 layers.
	trainable_parameters = [
		parameter for parameter in model.parameters() if parameter.requires_grad
	]
# автоматично адаптує швидкість навчання для кожного параметра;
# добре підходить як базовий оптимізатор;
# зазвичай швидше сходиться, ніж звичайний SGD;
# для ResNet оновлює лише незаморожений фінальний шар.
	# Adam adapts the learning rate for each parameter and is a practical baseline optimizer.
	optimizer = torch.optim.Adam(trainable_parameters, lr=0.001)
	return criterion, optimizer

# Навчання моделі:
def train_model(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	num_epochs: int = NUM_EPOCHS,
	checkpoint_name: str = "best_model.pt",
) -> dict[str, list[float]]:
	"""Train a model and save its weights when validation loss improves."""
	# Use a GPU when available; the same training code also works on the CPU.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
	checkpoint_path = CHECKPOINT_DIRECTORY / checkpoint_name
	best_validation_loss = float("inf")
	history = {"train_loss": [], "val_loss": []}

	for epoch in range(num_epochs):
		# Training mode enables gradient updates and training-specific layer behavior.
		model.train()
		train_loss = 0.0

		for images, labels in train_loader:
			# Inputs and labels must be on the same device as the model.
			images, labels = images.to(device), labels.to(device)
			# Remove gradients left by the previous batch.
			optimizer.zero_grad()
			# Forward pass: calculate one score for every class.
			outputs = model(images)
			loss = criterion(outputs, labels)
			# Backward pass: calculate gradients for all trainable weights.
			loss.backward()
			# Update the weights using the calculated gradients.
			optimizer.step()
			train_loss += loss.item() * images.size(0)

		# Evaluation mode disables training-only behavior such as dropout.
		model.eval()
		validation_loss = 0.0
		# Validation measures performance without changing weights.
		with torch.no_grad():
			for images, labels in val_loader:
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				validation_loss += loss.item() * images.size(0)

		average_train_loss = train_loss / len(train_loader.dataset)
		average_validation_loss = validation_loss / len(val_loader.dataset)
		history["train_loss"].append(average_train_loss)
		history["val_loss"].append(average_validation_loss)

		# Keep the checkpoint with the lowest validation loss seen so far.
		if average_validation_loss < best_validation_loss:
			best_validation_loss = average_validation_loss
			torch.save(model.state_dict(), checkpoint_path)
			print(f"Saved best model to: {checkpoint_path}")

		print(
			f"Epoch {epoch + 1}/{num_epochs}: "
			f"train_loss={average_train_loss:.4f}, "
			f"val_loss={average_validation_loss:.4f}"
		)

	return history


def evaluate_model(
	model: nn.Module,
	test_loader: DataLoader,
	checkpoint_name: str,
) -> tuple[dict[str, float], list[int], list[int]]:
	"""Evaluate the best saved model on unseen test images."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint_path = CHECKPOINT_DIRECTORY / checkpoint_name
	model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	model.to(device)
	model.eval()

	all_predictions = []
	all_labels = []
	# Test data is used only after training and is never used to update weights.
	with torch.no_grad():
		for images, labels in test_loader:
			outputs = model(images.to(device))
			predictions = outputs.argmax(dim=1).cpu()
			all_predictions.extend(predictions.tolist())
			all_labels.extend(labels.tolist())

	# Macro F1 gives every class equal importance, including less frequent classes.
	macro_f1 = f1_score(all_labels, all_predictions, average="macro")
	metrics = {
		"accuracy": accuracy_score(all_labels, all_predictions),
		"macro_f1": macro_f1,
		"weighted_f1": f1_score(
			all_labels,
			all_predictions,
			average="weighted",
		),
	}
	print(f"{checkpoint_name} test metrics: {metrics}")
	return metrics, all_labels, all_predictions


def plot_training_history(
	history: dict[str, list[float]],
	model_name: str,
) -> None:
	"""Save train and validation loss curves for one model."""
	REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 5))
	plt.plot(history["train_loss"], label="train loss")
	plt.plot(history["val_loss"], label="validation loss")
	plt.xlabel("Epoch")
	plt.ylabel("Cross-entropy loss")
	plt.title(f"Learning curves: {model_name}")
	plt.legend()
	plt.tight_layout()
	plt.savefig(REPORT_DIRECTORY / f"{model_name}_learning_curves.png")
	plt.close()


def analyze_classification_errors(
	labels: list[int],
	predictions: list[int],
	class_names: list[str],
	model_name: str,
) -> None:
	"""Save a confusion matrix and report the most common class confusions."""
	REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
	matrix = confusion_matrix(labels, predictions)

	# A confusion matrix shows true classes by row and predicted classes by column.
	figure, axis = plt.subplots(figsize=(8, 8))
	image = axis.imshow(matrix, cmap="Blues")
	figure.colorbar(image, ax=axis)
	axis.set(
		xlabel="Predicted class",
		ylabel="True class",
		title=f"Confusion matrix: {model_name}",
		xticks=range(len(class_names)),
		yticks=range(len(class_names)),
		xticklabels=class_names,
		yticklabels=class_names,
	)
	for row in range(matrix.shape[0]):
		for column in range(matrix.shape[1]):
			axis.text(column, row, matrix[row, column], ha="center", va="center")
	figure.autofmt_xdate(rotation=45)
	figure.tight_layout()
	figure.savefig(REPORT_DIRECTORY / f"{model_name}_confusion_matrix.png")
	plt.close(figure)

	incorrect = sum(label != prediction for label, prediction in zip(labels, predictions))
	print(f"{model_name}: {incorrect} classification errors")
	for true_class in range(matrix.shape[0]):
		for predicted_class in range(matrix.shape[1]):
			if true_class != predicted_class and matrix[true_class, predicted_class] > 0:
				print(
					f"  {class_names[true_class]} -> "
					f"{class_names[predicted_class]}: "
					f"{matrix[true_class, predicted_class]}"
				)

if __name__ == "__main__":
	dataset_path = download_dataset()
	split_dataset(dataset_path)
	train_loader, val_loader, test_loader = create_dataloaders()
	# Both model variants must use the same number of output classes.
	model = SimpleCNN(num_classes=len(train_loader.dataset.classes))
	print(model)
	transfer_model = create_transfer_model(
		num_classes=len(train_loader.dataset.classes)
	)
	print(transfer_model)
	criterion, optimizer = create_training_components(model)
	transfer_criterion, transfer_optimizer = create_training_components(
		transfer_model
	)
	# Train the custom CNN and save its best validation checkpoint.
	simple_history = train_model(
		model,
		train_loader,
		val_loader,
		criterion,
		optimizer,
		checkpoint_name="best_simple_cnn.pt",
	)
	# Train ResNet18 independently and save its best validation checkpoint.
	transfer_history = train_model(
		transfer_model,
		train_loader,
		val_loader,
		transfer_criterion,
		transfer_optimizer,
		checkpoint_name="best_resnet18.pt",
	)
	# Evaluate only the best checkpoint for each model on the untouched test set.
	simple_metrics, simple_labels, simple_predictions = evaluate_model(
		model, test_loader, "best_simple_cnn.pt"
	)
	transfer_metrics, transfer_labels, transfer_predictions = evaluate_model(
		transfer_model, test_loader, "best_resnet18.pt"
	)
	class_names = train_loader.dataset.classes
	plot_training_history(simple_history, "simple_cnn")
	plot_training_history(transfer_history, "resnet18")
	analyze_classification_errors(
		simple_labels,
		simple_predictions,
		class_names,
		"simple_cnn",
	)
	analyze_classification_errors(
		transfer_labels,
		transfer_predictions,
		class_names,
		"resnet18",
	)

    