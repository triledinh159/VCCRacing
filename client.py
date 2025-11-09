"""
Autonomous Vehicle Client

A client for connecting to and controlling an autonomous vehicle simulation,
retrieving state data and camera images (raw and segmented).
"""

import socket
import json
import argparse
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class CommandMode:
    """Command mode constants for vehicle communication."""
    STATE_DATA = 185  # Request vehicle state data
    RAW_IMAGE = 203   # Request raw camera image
    SEGMENTED_IMAGE = 31  # Request segmented camera image


class AVClientError(Exception):
    """Base exception for AV Client errors."""
    pass


class ConnectionError(AVClientError):
    """Raised when connection to server fails."""
    pass


class DataRetrievalError(AVClientError):
    """Raised when data retrieval fails."""
    pass


class AVClient:
    """
    Client for autonomous vehicle simulation control and data retrieval.

    Manages connection to vehicle server, sends control commands,
    and retrieves state data and camera images.
    """

    MAX_DGRAM = 2**16  # Maximum datagram size for socket operations

    def __init__(self, host: str = '127.0.0.1', port: int = 11000, timeout: float = 5.0):
        """
        Initialize AV Client.

        Args:
            host: Server hostname or IP address
            port: Server port number
            timeout: Socket timeout in seconds (default: 5.0)
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.speed_cmd = 0
        self.angle_cmd = 0

    def connect(self) -> None:
        """
        Establish connection to vehicle server.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            print(f"Connected to vehicle server at {self.host}:{self.port}")
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")

    def disconnect(self) -> None:
        """Close connection to vehicle server."""
        if self.socket:
            try:
                self.socket.close()
                print("Disconnected from vehicle server")
            except socket.error as e:
                print(f"Error closing socket: {e}")
            finally:
                self.socket = None

    def __enter__(self):
        """Context manager entry - establish connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up connection."""
        self.disconnect()
        return False

    def set_control(self, speed: int, angle: int) -> None:
        """
        Set vehicle control commands.

        Args:
            speed: Speed command (max 90)
            angle: Steering angle command (max ±25)
        """
        self.speed_cmd = speed
        self.angle_cmd = angle

    def _create_command_json(self, mode: int) -> bytes:
        """
        Create JSON command for server communication.

        Args:
            mode: Command mode (STATE_DATA, RAW_IMAGE, or SEGMENTED_IMAGE)

        Returns:
            JSON command as bytes
        """
        cmd: Dict[str, Any] = {'Cmd': mode}

        if mode == CommandMode.STATE_DATA:
            cmd['Speed'] = self.speed_cmd
            cmd['Angle'] = self.angle_cmd

        return json.dumps(cmd).encode('utf-8')

    def _receive_data(self, expected_length: int) -> bytes:
        """
        Receive data from socket with length validation.

        Args:
            expected_length: Expected number of bytes to receive

        Returns:
            Received data bytes

        Raises:
            DataRetrievalError: If data reception fails
        """
        if not self.socket:
            raise DataRetrievalError("Not connected to server")

        data = b''
        try:
            while len(data) < expected_length:
                to_read = expected_length - len(data)
                chunk = self.socket.recv(
                    min(self.MAX_DGRAM, to_read)
                )
                if not chunk:
                    raise DataRetrievalError("Connection closed by server")
                data += chunk
        except socket.error as e:
            raise DataRetrievalError(f"Socket error during data reception: {e}")

        return data

    def get_state_data(self) -> Dict[str, Any]:
        """
        Retrieve current vehicle state data.

        Returns:
            Dictionary containing vehicle state information

        Raises:
            DataRetrievalError: If state data retrieval fails
        """
        if not self.socket:
            raise DataRetrievalError("Not connected to server")

        try:
            # Send state data request
            print("[DEBUG] Sending state data request...")
            self.socket.sendall(self._create_command_json(CommandMode.STATE_DATA))

            # Receive data length
            print("[DEBUG] Waiting for data length...")
            length_bytes = self.socket.recv(8)
            if len(length_bytes) != 8:
                raise DataRetrievalError("Failed to receive data length")

            data_length = int.from_bytes(length_bytes, "big")
            print(f"[DEBUG] Expecting {data_length} bytes of state data")

            # Receive state data
            data = self._receive_data(data_length)
            print("[DEBUG] State data received successfully")
            return json.loads(data)

        except socket.timeout:
            raise DataRetrievalError("Timeout waiting for state data from server")
        except json.JSONDecodeError as e:
            raise DataRetrievalError(f"Invalid JSON response: {e}")
        except socket.error as e:
            raise DataRetrievalError(f"Socket error: {e}")

    def get_image(self, mode: int) -> np.ndarray:
        """
        Retrieve camera image from vehicle.

        Args:
            mode: Image mode (RAW_IMAGE or SEGMENTED_IMAGE)

        Returns:
            Image as numpy array

        Raises:
            DataRetrievalError: If image retrieval fails
            ValueError: If invalid mode specified
        """
        if mode not in (CommandMode.RAW_IMAGE, CommandMode.SEGMENTED_IMAGE):
            raise ValueError(f"Invalid image mode: {mode}")

        if not self.socket:
            raise DataRetrievalError("Not connected to server")

        image_type = "raw" if mode == CommandMode.RAW_IMAGE else "segmented"

        try:
            # Send image request
            print(f"[DEBUG] Sending {image_type} image request...")
            self.socket.sendall(self._create_command_json(mode))

            # Receive data length
            print(f"[DEBUG] Waiting for {image_type} image length...")
            length_bytes = self.socket.recv(8)
            if len(length_bytes) != 8:
                raise DataRetrievalError("Failed to receive data length")

            data_length = int.from_bytes(length_bytes, "big")
            print(f"[DEBUG] Expecting {data_length} bytes of {image_type} image data")

            # Receive image data
            data = self._receive_data(data_length)

            # Decode image
            print(f"[DEBUG] Decoding {image_type} image...")
            image = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_UNCHANGED
            )

            if image is None:
                raise DataRetrievalError("Failed to decode image data")

            print(f"[DEBUG] {image_type.capitalize()} image received successfully")
            return image

        except socket.timeout:
            raise DataRetrievalError(f"Timeout waiting for {image_type} image from server")
        except socket.error as e:
            raise DataRetrievalError(f"Socket error: {e}")

    def get_raw_image(self) -> np.ndarray:
        """
        Retrieve raw camera image.

        Returns:
            Raw camera image as numpy array
        """
        return self.get_image(CommandMode.RAW_IMAGE)

    def get_segmented_image(self) -> np.ndarray:
        """
        Retrieve segmented camera image.

        Returns:
            Segmented camera image as numpy array
        """
        return self.get_image(CommandMode.SEGMENTED_IMAGE)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Autonomous Vehicle Client - Connect to AV simulation server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py                    # Connect to default 127.0.0.1:11000
  python client.py 11001              # Connect to 127.0.0.1:11001
  python client.py --host 192.168.1.10 --port 11000
  python client.py --timeout 10       # Set 10 second timeout
        """
    )

    parser.add_argument(
        'port',
        nargs='?',
        type=int,
        default=11000,
        help='Server port (default: 11000)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server hostname or IP address (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='Socket timeout in seconds (default: 5.0)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Disable debug messages'
    )

    return parser.parse_args()


def main():
    """Main execution loop for AV client demonstration."""
    args = parse_arguments()

    print(f"[CONFIG] Connecting to {args.host}:{args.port} (timeout: {args.timeout}s)")

    try:
        with AVClient(host=args.host, port=args.port, timeout=args.timeout) as client:
            print("[INFO] Starting main loop. Press 'q' to exit.\n")
            while True:
                print("[LOOP] Fetching vehicle data...")

                # Your application logic here

                # Retrieve vehicle data
                state = client.get_state_data()
                raw_image = client.get_raw_image()
                segmented_image = client.get_segmented_image()

                # Display data
                print(f"[DATA] State: {state}")
                cv2.imshow('Raw Camera', raw_image)
                cv2.imshow('Segmented Camera', segmented_image)

                # Send control commands (max speed: 90, max angle: ±25)
                client.set_control(speed=10, angle=0)

                # Exit on 'q' key
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("[INFO] Exit requested by user")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except AVClientError as e:
        print(f"AV Client error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
