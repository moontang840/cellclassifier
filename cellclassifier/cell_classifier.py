import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

class CentroidClassifier:
    def __init__(self, normalize=True):
        """
        Nearest Centroid Classifier
        
        Parameters:
        normalize: Whether to standardize features
        """
        self.normalize = normalize
        self.centroids = {}
        self.scaler = StandardScaler() if normalize else None
        self.cell_types = ['LX-2', 'Hep3B', 'HepG2', 'Huh-7', 'MHCC97H']
        self.markers = ['miR-141', 'miR-155', 'miR-21', 'miR-221', 'miR-222']
        
    def load_data(self, data_dir='data'):
        """
        Load all marker data
        """
        data_dict = {}
        
        for marker in self.markers:
            file_path = Path(data_dir) / f'{marker}.csv'
            df = pd.read_csv(file_path)
            
            # Clean column names (remove BOM)
            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
            data_dict[marker] = df
        return data_dict

    def fit_cv(self, X, y, n_splits=5, random_state=42):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.cv_centroids = []  # 用来存每折质心
        cv_scores = []
        for fold, (tr, val) in enumerate(kf.split(X)):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")
            fold_clf = CentroidClassifier(normalize=self.normalize)
            fold_clf.fit(X[tr], y[tr])
            score = (fold_clf.predict(X[val]) == y[val]).mean()
            cv_scores.append(score)
            self.cv_centroids.append({
                'fold': fold + 1,
                'centroids': fold_clf.get_centroids(),
                'score': score
            })
            print(f"Fold {fold + 1} accuracy: {score:.4f}")
        return cv_scores

    # =============== 1. 批次平均训练 ===============
    def fit_batch_average(self, X, y, batch_size=10, random_state=42):
        """用批次平均向量重新计算质心（训练阶段）"""
        if self.normalize:
            X = self.scaler.fit_transform(X)

        rng = np.random.default_rng(random_state)
        for cell_type in self.cell_types:
            mask = y == cell_type
            X_cell = X[mask]
            if len(X_cell) == 0:
                continue
            real_batch = min(batch_size, len(X_cell))
            idx = rng.choice(len(X_cell), real_batch, replace=False)
            avg_vec = X_cell[idx].mean(axis=0)
            self.centroids[cell_type] = avg_vec
        print(f"Batch-average training done (batch_size={batch_size})!")
        return self

    # =============== 2. 批次平均交叉验证 ===============
    def fit_cv_batch(self, X, y, n_splits=5, batch_size=10, random_state=42):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.cv_centroids = []
        cv_scores = []

        for fold, (tr, val) in enumerate(kf.split(X)):
            print(f"\n=== Fold {fold + 1}/{n_splits} (batch_size={batch_size}) ===")
            fold_clf = CentroidClassifier(normalize=self.normalize)
            # 关键：用批次平均训练
            fold_clf.fit_batch_average(X[tr], y[tr], batch_size=batch_size)
            # 验证时同样用批次平均预测
            val_pred = []
            val_true = []
            for cell_type in self.cell_types:
                mask = y[val] == cell_type
                X_cell = X[val][mask]
                if len(X_cell) == 0:
                    continue
                # 验证集也做同样批次平均
                real_batch = min(batch_size, len(X_cell))
                rng = np.random.default_rng(random_state + fold)
                idx = rng.choice(len(X_cell), real_batch, replace=False)
                avg_vec = X_cell[idx].mean(axis=0, keepdims=True)
                pred = fold_clf.predict(avg_vec)[0]
                val_pred.append(pred)
                val_true.append(cell_type)

            acc = accuracy_score(val_true, val_pred)
            cv_scores.append(acc)
            self.cv_centroids.append({
                'fold': fold + 1,
                'centroids': fold_clf.get_centroids(),
                'score': acc
            })
            print(f"Fold {fold + 1} batch-acc: {acc:.4f}")
        return cv_scores

    def show_all_folds_centroids(self):
        if not self.cv_centroids:
            raise RuntimeError("请先运行 fit_cv()")
        for fd in self.cv_centroids:
            print(f"\n=== Fold {fd['fold']}  (acc={fd['score']:.4f}) ===")
            print(fd['centroids'].round(3))

    def plot_all_folds_centroids(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        n = len(self.cv_centroids)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
        if n == 1: axes = [axes]
        for ax, fd in zip(axes, self.cv_centroids):
            sns.heatmap(fd['centroids'], annot=True, fmt=".2f", cmap="viridis", ax=ax)
            ax.set_title(f"Fold {fd['fold']}  (acc={fd['score']:.4f})")
        plt.tight_layout();
        plt.show()

        return data_dict
    
    def prepare_dataset(self, data_dict):
        """
        Convert data to machine learning format
        Ensure proper correspondence of 5 marker data for the same cell
        Returns: X (feature matrix), y (labels)
        """
        X = []
        y = []
        
        # First check data integrity and consistency
        print("Data integrity check:")
        for cell_type in self.cell_types:
            sample_counts = []
            for marker in self.markers:
                count = len(data_dict[marker][cell_type].dropna())
                sample_counts.append(count)
            print(f"{cell_type}: {dict(zip(self.markers, sample_counts))}")
        
        # 为每个细胞类型创建特征向量
        for cell_type in self.cell_types:
            # 获取该细胞类型在所有标志物中的数据
            cell_data = {}
            for marker in self.markers:
                cell_data[marker] = data_dict[marker][cell_type].dropna().values
            
            # Find minimum sample count (ensure consistent sample counts across all markers)
            min_samples = min(len(values) for values in cell_data.values())
            
            # Warning for data imbalance
            max_samples = max(len(values) for values in cell_data.values())
            if min_samples != max_samples:
                print(f"Warning: Marker data counts for {cell_type} are inconsistent, using first {min_samples} samples")
            
            # Create feature vectors - ensure same row index corresponds to same cell
            for i in range(min_samples):
                feature_vector = []
                for marker in self.markers:  # Keep marker order consistent
                    feature_vector.append(cell_data[marker][i])
                X.append(feature_vector)
                y.append(cell_type)
        
        X = np.array(X)
        y = np.array(y)
        
        # Dataset statistics
        print(f"\nDataset statistics:")
        print(f"Total samples: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        unique, counts = np.unique(y, return_counts=True)
        for cell_type, count in zip(unique, counts):
            print(f"{cell_type}: {count} samples")
        
        # Check data balance
        if len(set(counts)) > 1:
            print("Warning: Data imbalance may affect classification performance")
        
        return X, y
    
    def fit(self, X, y):
        """
        Train classifier: Calculate centroids for each cell type
        """
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        # Calculate centroids for each cell type
        for cell_type in self.cell_types:
            mask = y == cell_type
            if np.any(mask):
                self.centroids[cell_type] = np.mean(X[mask], axis=0)
        
        print(f"Training completed! Calculated centroids for {len(self.centroids)} cell types")
        return self
    
    def predict(self, X):
        """
        Predict cell types for new samples
        """
        if self.normalize:
            X = self.scaler.transform(X)
        
        predictions = []
        
        for sample in X:
            # Calculate distance to each centroid
            distances = {}
            for cell_type, centroid in self.centroids.items():
                distance = np.linalg.norm(sample - centroid)
                distances[cell_type] = distance
            
            # Select the nearest centroid
            predicted_cell = min(distances, key=distances.get)
            predictions.append(predicted_cell)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Return prediction probabilities (based on inverse distance)
        """
        if self.normalize:
            X = self.scaler.transform(X)
        
        probabilities = []
        
        for sample in X:
            distances = {}
            for cell_type, centroid in self.centroids.items():
                distance = np.linalg.norm(sample - centroid)
                distances[cell_type] = distance
            
            # Convert distances to probabilities (smaller distance = higher probability)
            inv_distances = {cell: 1/(dist + 1e-10) for cell, dist in distances.items()}
            total = sum(inv_distances.values())
            probs = {cell: prob/total for cell, prob in inv_distances.items()}
            
            probabilities.append([probs[cell] for cell in self.cell_types])
        
        return np.array(probabilities)
    
    def get_centroids(self):
        """
        Get centroid information
        """
        centroids_df = pd.DataFrame(self.centroids).T
        centroids_df.columns = self.markers
        return centroids_df
    
    def plot_centroids(self):
        """
        Visualize centroids
        """
        centroids_df = self.get_centroids()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(centroids_df, annot=True, cmap='viridis', 
                   fmt='.2f', cbar_kws={'label': 'Marker Expression Level'})
        plt.title('Cell Type Centroid Heatmap')
        plt.xlabel('miRNA Markers')
        plt.ylabel('Cell Types')
        plt.tight_layout()
        plt.show()
    
    def evaluate_batch_prediction(self, X_test, y_test):
        """
        Evaluate model using batch average prediction (simulating real experimental conditions)
        Each cell type's test samples are averaged to simulate batch testing
        """
        batch_predictions = []
        batch_actuals = []
        batch_details = []
        
        print("\nBatch Average Prediction Results:")
        print("=" * 40)
        
        for cell_type in self.cell_types:
            # Get all test samples for this cell type
            mask = y_test == cell_type
            if np.any(mask):
                cell_samples = X_test[mask]
                # Calculate average features across all samples of this cell type
                avg_features = np.mean(cell_samples, axis=0)
                
                # Predict using the average feature vector
                prediction = self.predict([avg_features])[0]
                probabilities = self.predict_proba([avg_features])[0]
                
                batch_predictions.append(prediction)
                batch_actuals.append(cell_type)
                
                # Store details for reporting
                batch_details.append({
                    'cell_type': cell_type,
                    'n_samples': len(cell_samples),
                    'avg_features': avg_features,
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'correct': prediction == cell_type
                })
                
                print(f"{cell_type} (n={len(cell_samples)}):")
                print(f"  Average features: {[f'{x:.1f}' for x in avg_features]}")
                print(f"  Predicted: {prediction}")
                print(f"  Probabilities: {dict(zip(self.cell_types, [f'{p:.3f}' for p in probabilities]))}")
                print(f"  Result: {'✓ Correct' if prediction == cell_type else '✗ Incorrect'}")
                print()
        
        # Calculate batch accuracy
        batch_accuracy = accuracy_score(batch_actuals, batch_predictions)
        print(f"Batch prediction accuracy: {batch_accuracy:.4f} ({sum(1 for d in batch_details if d['correct'])}/{len(batch_details)})")
        
        return {
            'batch_accuracy': batch_accuracy,
            'batch_predictions': batch_predictions,
            'batch_actuals': batch_actuals,
            'batch_details': batch_details
        }
    
    def evaluate_model(self, X, y, cv_folds=5, test_size=0.2, random_state=42):
        """
        Evaluate model performance with both individual cell prediction and batch average prediction
        """
        print("=" * 50)
        print("Model Performance Evaluation")
        print("=" * 50)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model on training set
        self.fit(X_train, y_train)
        
        # Batch average prediction (realistic experimental method)
        print("\n" + "=" * 30)
        print("Batch Average Prediction (Realistic)")
        print("=" * 30)
        
        batch_results = self.evaluate_batch_prediction(X_test, y_test)
        
        # Cross-validation for batch prediction
        print("\n" + "=" * 30)
        print("Cross-validation (Batch Average)")
        print("=" * 30)
        
        cv_batch_accuracies = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Train on fold
            cv_classifier = CentroidClassifier(normalize=self.normalize)
            cv_classifier.fit(X_train_cv, y_train_cv)
            
            # Evaluate using batch average method
            fold_batch_results = cv_classifier.evaluate_batch_prediction(X_val_cv, y_val_cv)
            cv_batch_accuracies.append(fold_batch_results['batch_accuracy'])
            
            print(f"Fold {fold+1} batch accuracy: {fold_batch_results['batch_accuracy']:.4f}")
        
        cv_batch_mean = np.mean(cv_batch_accuracies)
        cv_batch_std = np.std(cv_batch_accuracies)
        
        print(f"\nBatch CV average accuracy: {cv_batch_mean:.4f} ± {cv_batch_std:.4f}")
        print(f"Batch CV scores: {cv_batch_accuracies}")
        
        return {
            'batch_accuracy': batch_results['batch_accuracy'],
            'batch_cv_scores': cv_batch_accuracies,
            'batch_cv_mean': cv_batch_mean,
            'batch_cv_std': cv_batch_std,
            'batch_details': batch_results['batch_details']
        }
    
    def analyze_batch_size_effect(self, X, y, batch_sizes=None, num_trials=20, test_size=0.2, random_state=42):
        """
        Analyze the effect of batch size on prediction accuracy
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 5, 10, 20, 50, 100]
        
        print("=" * 60)
        print("Batch Size Effect Analysis")
        print("=" * 60)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.fit(X_train, y_train)
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            trial_accuracies = []
            
            for trial in range(num_trials):
                batch_predictions = []
                batch_actuals = []
                
                for cell_type in self.cell_types:
                    # Get test samples for this cell type
                    mask = y_test == cell_type
                    cell_samples = X_test[mask]
                    
                    if len(cell_samples) >= batch_size:
                        # Randomly sample batch_size cells
                        indices = np.random.choice(len(cell_samples), batch_size, replace=False)
                        batch_samples = cell_samples[indices]
                        
                        # Calculate average
                        avg_features = np.mean(batch_samples, axis=0)
                        
                        # Predict
                        prediction = self.predict([avg_features])[0]
                        batch_predictions.append(prediction)
                        batch_actuals.append(cell_type)
                
                # Calculate accuracy for this trial
                if len(batch_predictions) > 0:
                    trial_accuracy = accuracy_score(batch_actuals, batch_predictions)
                    trial_accuracies.append(trial_accuracy)
            
            # Calculate statistics
            mean_accuracy = np.mean(trial_accuracies)
            std_accuracy = np.std(trial_accuracies)
            
            results.append({
                'batch_size': batch_size,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'num_trials': len(trial_accuracies)
            })
            
            print(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Display summary
        print("\n" + "=" * 40)
        print("Batch Size Effect Summary")
        print("=" * 40)
        print(f"{'Batch Size':<12} {'Accuracy':<12} {'Std Dev':<12} {'Trials':<8}")
        print("-" * 45)
        
        for result in results:
            print(f"{result['batch_size']:<12} {result['mean_accuracy']:<12.4f} {result['std_accuracy']:<12.4f} {result['num_trials']:<8}")
        
        # Find minimum batch size for >95% accuracy
        high_accuracy_batches = [r for r in results if r['mean_accuracy'] >= 0.95]
        if high_accuracy_batches:
            min_batch_95 = min(high_accuracy_batches, key=lambda x: x['batch_size'])
            print(f"\nMinimum batch size for >95% accuracy: {min_batch_95['batch_size']}")
        
        # Find minimum batch size for >99% accuracy
        very_high_accuracy_batches = [r for r in results if r['mean_accuracy'] >= 0.99]
        if very_high_accuracy_batches:
            min_batch_99 = min(very_high_accuracy_batches, key=lambda x: x['batch_size'])
            print(f"Minimum batch size for >99% accuracy: {min_batch_99['batch_size']}")
        
        return results

def main():
    # Create classifier
    classifier = CentroidClassifier(normalize=True)
    
    # Load data
    print("Loading data...")
    data_dict = classifier.load_data()
    
    # Prepare dataset
    print("\nPreparing dataset...")
    X, y = classifier.prepare_dataset(data_dict)

    # 3. 交叉验证 + 批次平均质心（逐折打印）
    print("\n" + "=" * 60)
    print("5-Fold CV – 批次平均质心 (batch_size=10)")
    print("=" * 60)
    cv_scores = classifier.fit_cv_batch(X, y, n_splits=5, batch_size=100)
    classifier.show_all_folds_centroids()  # 已打印每折质心数值

    # 4. 用全量数据训练最终模型并评估
    print("\nFinal model evaluation (full data)...")
    results = classifier.evaluate_model(X, y)

    # 5. 批量大小影响分析
    print("\n" + "=" * 50)
    print("Batch Size Effect Analysis")
    print("=" * 50)
    batch_size_results = classifier.analyze_batch_size_effect(X, y)

    # 6. 返回给外层
    return classifier, results, batch_size_results

if __name__ == "__main__":
    classifier, results, batch_size_results = main()
