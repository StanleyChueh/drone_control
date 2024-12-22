import cv2
import numpy as np
from pupil_apriltags import Detector
from djitellopy import Tello
import math

class TelloCamera:
    def __init__(self, drone, camera_matrix=None, dist_coeffs=None, tag_size=0.16):
        # Use the provided Tello drone object
        self.drone = drone
        self.drone.streamon()  # Start video stream
        self.cap = self.drone.get_frame_read()  # Get the video feed from the drone

        # Initialize AprilTag detector
        self.at_detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0, quad_sigma=0.0)

        # Camera intrinsic parameters
        # Allow passing custom calibration parameters; use defaults if none are provided
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(4, dtype=np.float64)

        # Set the size of the AprilTag (in meters)
        self.tag_size = tag_size

    def project_3d_points(self, points_3d, rvec, tvec):
        # Project 3D points onto the image plane
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        return points_2d.reshape(-1, 2)
    
    def draw_cube(self, img, rvec, tvec, tag_size):
        # Define the 3D points of the cube
        half_size = tag_size / 2
        cube_points_3d = np.array([
            [-half_size, -half_size, 0],  # Bottom face
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0],
            [-half_size, -half_size, -tag_size],  # Top face
            [half_size, -half_size, -tag_size],
            [half_size, half_size, -tag_size],
            [-half_size, half_size, -tag_size]
        ], dtype=np.float32)

        # Project the cube points to the image plane
        projected_points = self.project_3d_points(cube_points_3d, rvec, tvec)

        # Draw cube edges
        projected_points = np.int32(projected_points)
        bottom_face = projected_points[:4]
        top_face = projected_points[4:]

        # Bottom face edges
        cv2.polylines(img, [bottom_face], isClosed=True, color=(0, 255, 0), thickness=2)

        # Top face edges
        cv2.polylines(img, [top_face], isClosed=True, color=(0, 255, 0), thickness=2)

        # Vertical edges
        for i in range(4):
            cv2.line(img, tuple(bottom_face[i]), tuple(top_face[i]), (0, 255, 0), 2)

    def run(self):
        # Get the current frame
        img = self.cap.frame
        if img is None:
            print("No frame received from Tello.")
            return None  # Return None if no frame is available

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        tags = self.at_detector.detect(
            gray, estimate_tag_pose=True, camera_params=(
                self.camera_matrix[0, 0],  # fx
                self.camera_matrix[1, 1],  # fy
                self.camera_matrix[0, 2],  # cx
                self.camera_matrix[1, 2]   # cy
            ),
            tag_size=self.tag_size
        )

        for tag in tags:
            # Draw the detected tag on the image
            (cX, cY) = tag.center
            cv2.circle(img, (int(cX), int(cY)), 10, (0, 255, 0), -1)
            cv2.putText(img, f"ID: {tag.tag_id}", (int(cX), int(cY) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw a square around the tag
            corners = np.int0(tag.corners)
            cv2.polylines(img, [corners], isClosed=True, color=(255, 0, 0), thickness=2)

            # Pose estimation
            if hasattr(tag, 'pose_R') and hasattr(tag, 'pose_t'):
                rvec, _ = cv2.Rodrigues(np.array(tag.pose_R))  # Convert rotation matrix to Rodrigues
                tvec = np.array(tag.pose_t).reshape(-1, 1)
                
                # Draw the cube on the tag
                self.draw_cube(img, rvec, tvec, self.tag_size)
                
                # Define 3D axes for visualization
                axis_length = self.tag_size / 2
                axes_points_3d = np.array([
                    [0, 0, 0],                    # Origin
                    [axis_length, 0, 0],          # X-axis
                    [0, axis_length, 0],          # Y-axis
                    [0, 0, -axis_length]          # Z-axis
                ])

                # Project 3D axes points onto the image plane
                projected_axes = self.project_3d_points(axes_points_3d, rvec, tvec)

                # Draw the axes
                origin = tuple(np.int32(projected_axes[0]))
                x_axis = tuple(np.int32(projected_axes[1]))
                y_axis = tuple(np.int32(projected_axes[2]))
                z_axis = tuple(np.int32(projected_axes[3]))

                cv2.arrowedLine(img, origin, x_axis, (0, 0, 255), 2)  # X-axis (Red)
                cv2.arrowedLine(img, origin, y_axis, (0, 255, 0), 2)  # Y-axis (Green)
                cv2.arrowedLine(img, origin, z_axis, (255, 0, 0), 2)  # Z-axis (Blue)

                # Display pitch, yaw, roll
                r = np.array(tag.pose_R)
                pitch = math.atan2(r[2, 1], r[2, 2])
                yaw = math.atan2(-r[2, 0], math.sqrt(r[2, 1]**2 + r[2, 2]**2))
                roll = math.atan2(r[1, 0], r[0, 0])

                cv2.putText(img, f"Pitch: {math.degrees(pitch):.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"Yaw: {math.degrees(yaw):.2f}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"Roll: {math.degrees(roll):.2f}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

if __name__ == "__main__":
    try:
        mydrone = Tello()
        mydrone.connect()
        print(f"Battery: {mydrone.get_battery()}%")
        tello_camera = TelloCamera(mydrone)

        while True:
            img = tello_camera.run()
            if img is None:
                continue

            cv2.imshow("Tello Camera Stream", img)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        cv2.destroyAllWindows()
        mydrone.streamoff()


