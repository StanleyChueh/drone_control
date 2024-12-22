from djitellopy import Tello
from threading import Thread, Lock
from pynput import keyboard
import cv2
from tello_apriltag import TelloCamera

# Global variables for drone control
lr, fb, ud, yv = 0, 0, 0, 0
lock = Lock()  # Ensure thread safety for control variables


def getKeyboardInput(drone):
    global lr, fb, ud, yv

    def on_press(key):
        global lr, fb, ud, yv
        try:
            if key.char == 'e':  # Takeoff
                with lock:
                    drone.takeoff()
            elif key.char == 'q':  # Land
                with lock:
                    drone.land()
            elif key.char == 'w':  # Up
                with lock:
                    ud = 50
            elif key.char == 's':  # Down
                with lock:
                    ud = -50
            elif key.char == 'a':  # Rotate left
                with lock:
                    yv = -50
            elif key.char == 'd':  # Rotate right
                with lock:
                    yv = 50
        except AttributeError:
            pass

        # Movement keys
        try:
            if key == keyboard.Key.left:
                with lock:
                    lr = -50
            elif key == keyboard.Key.right:
                with lock:
                    lr = 50
            elif key == keyboard.Key.up:
                with lock:
                    fb = 50
            elif key == keyboard.Key.down:
                with lock:
                    fb = -50
        except AttributeError:
            pass

    def on_release(key):
        global lr, fb, ud, yv
        try:
            with lock:
                if key in [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down]:
                    lr, fb = 0, 0
                elif key.char in ['w', 's']:
                    ud = 0
                elif key.char in ['a', 'd']:
                    yv = 0
        except AttributeError:
            pass

    # Start the listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    try:
        # Initialize Tello drone
        mydrone = Tello()
        mydrone.connect()
        print(f"Battery: {mydrone.get_battery()}%")

        # Start keyboard control in a separate thread
        keyboard_thread = Thread(target=getKeyboardInput, args=(mydrone,))
        keyboard_thread.start()

        # Initialize AprilTag detection
        tello_camera = TelloCamera(mydrone)

        # Run AprilTag detection
        while True:
            # Detect and display AprilTags
            img = tello_camera.run()
            if img is None:  # Skip iteration if no frame is received
                continue
            
            cv2.imshow("Tello Camera Stream", img)

            # Send control commands to the drone
            with lock:
                mydrone.send_rc_control(lr, fb, ud, yv)

            # Quit the loop on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Clean up
        cv2.destroyAllWindows()
        mydrone.streamoff()
