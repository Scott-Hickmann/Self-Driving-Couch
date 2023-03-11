import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
from tracking import track_shirt
import matplotlib.pyplot as plt

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def start_processing_stream(self):
        x_path = []
        y_path = []
        z_path = []

        plt.ion()
        try:
            while True:
                self.event.wait()  # Wait for new frame to arrive

                # Copy the newly arrived RGBD frame
                depth = self.session.get_depth_frame()
                rgb = self.session.get_rgb_frame()
                intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
                camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

                # print(intrinsic_mat)

                # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

                # Postprocess it
                if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                    depth = cv2.flip(depth, 1)
                    rgb = cv2.flip(rgb, 1)

                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                rgb_with_shirt, cx, cy = track_shirt(rgb)

                sol = np.linalg.inv(intrinsic_mat) @ np.array([cx, cy, 1])
                z = depth[cy * depth.shape[0] // rgb.shape[0], cx * depth.shape[1] // rgb.shape[1]]
                x = (sol[0] * z) / sol[2]
                y = -(sol[1] * z) / sol[2]

                print(f"X: {x}, Y: {y}, Z: {z}")

                x_path.append(x)
                y_path.append(y)
                z_path.append(z)
                # Data for a three-dimensional line

                # Show the RGBD Stream
                cv2.imshow('RGB', rgb_with_shirt)
                cv2.imshow('Depth', depth)

                cv2.waitKey(1)
                plt.draw()

                # plt.plot(x_path, z_path, 'gray')
                # plt.show()
                M = 50
                N = 10
                x_moving_average = moving_average(x_path, N)
                plt.clf()
                plt.plot(x_moving_average, z_path[N - 1:], 'gray')
                plt.xlim([-1, 1])
                plt.ylim([0, 3])
                plt.show()
                x_path = x_path[-M:]
                y_path = y_path[-M:]
                z_path = z_path[-M:]

                self.event.clear()
        except KeyboardInterrupt:
            # ax = plt.axes(projection='3d')
            # ax.plot3D(x_path, y_path, z_path, 'gray')
            # plt.plot(x_path, z_path, 'gray')
            # plt.show()
            # N = 30
            # x_moving_average = moving_average(x_path, N)
            # plt.plot(x_moving_average, z_path[N - 1:], 'gray')
            # plt.show()
            pass


if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()