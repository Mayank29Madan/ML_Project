import os
import joblib

# Save function to store the trained model in a specific directory
def save_model(model, filename):
    """
    Save the trained model to a .pkl file in the 'save_model' directory.
    
    :param model: Trained model object (e.g., RandomForest, XGBoost, etc.)
    :param filename: The name of the file (without path) where the model will be saved
    """
    try:
        # Create 'save_model' directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), 'save_model')
        os.makedirs(save_dir, exist_ok=True)  # This creates the directory if it doesn't exist
        
        # Full path for saving the model
        save_path = os.path.join(save_dir, filename)
        
        # Save the model
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Error while saving the model: {e}")

def save_scaler(scaler, filename):
    """
    Save the trained model to a .pkl file in the 'save_model' directory.
    
    :param model: Trained model object (e.g., RandomForest, XGBoost, etc.)
    :param filename: The name of the file (without path) where the model will be saved
    """
    try:
        # Create 'save_model' directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), 'scalers')
        os.makedirs(save_dir, exist_ok=True)  # This creates the directory if it doesn't exist
        
        # Full path for saving the model
        save_path = os.path.join(save_dir, filename)
        
        # Save the model
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to {save_path}")
    except Exception as e:
        print(f"Error while saving the scaler: {e}")

# Load function to load a saved model from a .pkl file
def load_model(filename):
    """
    Load a trained model from the 'save_model' directory.
    
    :param filename: File path where the model is stored
    :return: Loaded model object
    """
    try:
        # Full path for loading the model
        save_dir = os.path.join(os.getcwd(), 'save_model')
        load_path = os.path.join(save_dir, filename)
        
        # Load the model
        model = joblib.load(load_path)
        print(f"Model loaded from {load_path}")
        return model
    except Exception as e:
        print(f"Error while loading the model: {e}")
        return None
    
# Load function to load a saved model from a .pkl file
def load_scaler(filename):
    """
    Load a trained model from the 'save_model' directory.
    
    :param filename: File path where the model is stored
    :return: Loaded model object
    """
    try:
        # Full path for loading the model
        save_dir = os.path.join(os.getcwd(), 'scalers')
        load_path = os.path.join(save_dir, filename)
        
        # Load the model
        scaler = joblib.load(load_path)
        print(f"Scaler loaded from {load_path}")
        return scaler
    except Exception as e:
        print(f"Error while loading the sclaer: {e}")
        return None
