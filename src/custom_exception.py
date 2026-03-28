import sys
import traceback
from typing import Optional


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information including
    filename, line number, and original exception details.
    """

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a detailed error message when the exception is printed."""
        return self.get_detailed_error_message()

    def get_detailed_error_message(self) -> str:
        """
        Generate a detailed error message with file, line number, and traceback info.
        """
        exc_type, exc_value, exc_tb = sys.exc_info()

        # If no active exception, return basic message
        if exc_tb is None:
            return f"Error: {self.message}"

        # Get detailed information
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_type = exc_type.__name__ if exc_type else "CustomException"

        # Build detailed message
        error_msg = [
            f"Error Type      : {error_type}",
            f"Error Message   : {self.message}",
            f"File            : {file_name}",
            f"Line            : {line_number}"
        ]

        # Add original exception details if available
        if self.original_exception:
            error_msg.append(f"Original Error  : {type(self.original_exception).__name__}: {self.original_exception}")

        # Add traceback summary (last few lines)
        tb_summary = traceback.format_exception_only(exc_type, exc_value)
        if tb_summary:
            error_msg.append(f"Traceback       : {tb_summary[-1].strip()}")

        return "\n".join(error_msg)

    def get_full_traceback(self) -> str:
        """Return the full traceback as a string (useful for logging)."""
        return "".join(traceback.format_exception(
            type(self), self, self.__traceback__
        ))


# Optional: Add a helper function for quick exception raising
def raise_custom_error(message: str, original_exception: Optional[Exception] = None):
    """Convenience function to raise CustomException."""
    raise CustomException(message, original_exception)


if __name__ == "__main__":
    # Test the custom exception
    try:
        raise CustomException("Failed to load model weights", ValueError("Invalid path"))
    except CustomException as e:
        print(e)
        print("\n--- Full Traceback ---")
        print(e.get_full_traceback())