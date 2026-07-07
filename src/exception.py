import sys
import mlflow

def error_message_detail(error, error_detail: sys):
    
    # Extract execution trace information
    _, _, exc_tb = error_detail.exc_info()

    # Fallback if traceback is unavailable
    if exc_tb is None:
        return f"Error occurred: {str(error)}"
    

    # Get the name of the Python file where the exception occurred.
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        "Error occurred in python script name [{0}] "
        "line number [{1}] "
        "error message [{2}]"
    ).format(file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys , log_to_mlflow: bool = True):
        # Call the constructor of the parent Exception class.
        super().__init__(error_message)

        # Store a detailed error message generated using the helper function above.
        self.error_message = error_message_detail(
            error_message, error_detail
        )

        # Automatically log to MLflow if the flag is True
        if log_to_mlflow:
            self._log_error_to_mlflow()

    def _log_error_to_mlflow(self):
        """Logs the exception details to the currently active MLflow run."""
        
        # Only attempt to log if an MLflow run is currently active
        if mlflow.active_run():
            try:
                # Log a status tag to easily filter failed runs in the MLflow UI
                mlflow.set_tag("pipeline_status", "FAILED")
                
                # Log a snippet of the error as a tag (MLflow tags have a 250 char limit)
                mlflow.set_tag("error_snippet", str(self.error_message)[:245] + "...")
                
                # Log the full detailed error trace as a text artifact
                mlflow.log_text(self.error_message, "error_logs/exception_trace.txt")
            except Exception as e:
                # Failsafe: print to console so MLflow failures don't mask the primary error
                print(f"Warning: Failed to log exception to MLflow. Details: {e}")

    def __str__(self):
        # Whenever the exception object is printed , return the detailed error message.
        return self.error_message