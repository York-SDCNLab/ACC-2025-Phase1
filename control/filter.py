import numpy as np

TWO_PI = 2 * np.pi

def wrap_to_pi(th: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi).

    Args:
        th (float): The angle to be wrapped in radians.

    Returns:
        float: The wrapped angle in radians.
    """
    th = th % TWO_PI
    th = (th + TWO_PI) % TWO_PI
    if th > np.pi:
        th -= TWO_PI
    return th

class EKF:
    """A class for performing Extended Kalman Filtering (EKF).

    Args:
        **kwargs: Additional keyword arguments.
        x_0 (ndarray): The initial state estimate as a 1D or 2D numpy array.
        P_0 (ndarray): The initial covariance matrix as a 2D numpy array.
        Q (ndarray): The process noise covariance matrix as a 2D numpy array.
        R (ndarray): The measurement noise covariance matrix as a 2D numpy
            array.
        A (ndarray): The state transition matrix as a 2D numpy array.
        B (ndarray): The control-input matrix as a 2D numpy array.
        f (function): The state transition function that maps the previous
            state to the next state.
        J_f (function): The Jacobian matrix of the state transition function.
        C (ndarray): The output matrix as a 2D numpy array.
        h (function): The measurement function that maps the state to the
            measurement space.
        J_h (function): The Jacobian matrix of the measurement function.

    Attributes:
        x_hat (ndarray): The current state estimate as a column vector.
        P (ndarray): The current covariance matrix.
        Q (ndarray): The process noise covariance matrix.
        R (ndarray): The measurement noise covariance matrix.
        A (ndarray): The state transition matrix.
        B (ndarray): The control-input matrix.
        f (function): The state transition function.
        J_f (function): The Jacobian matrix of the state transition function.
        C (ndarray): The output matrix.
        h (function): The measurement function.
        J_h (function): The Jacobian matrix of the measurement function.
        I (ndarray): The identity matrix.

    Raises:
        ValueError: If initial estimates `x_0` and `P_0` are not provided.
        ValueError: If noise covariance matrices `Q` and `R` are not provided.
        ValueError: If motion model is not properly defined.
        ValueError: If measurement model is not properly defined.
        ValueError: If noise covariance matrix `Q` has incorrect dimensions.
        ValueError: If output matrix `C` is not provided for linear
            measurement model.
        ValueError: If measurement function `h` and its Jacobian `J_h` are
            not provided for nonlinear measurement model.
        ValueError: If measurement model is defined using both `C` and
            `h`/`J_h`.
    """

    def __init__(self, **kwargs):
        """Initializes the EKF class instance with provided inputs. """
        #region : Load initial estimates and ensure their validity
        # Load initial state estimate and arrange it to be a column vector
        x_0 = kwargs.pop('x_0', None)
        if x_0 is None:
            raise ValueError(
                'Initial estimates x_0 and P_0 must both be provided.'
            )
        x_0 = np.squeeze(x_0)
        try:
            n = len(x_0)
        except TypeError:
            n = 1
        self.x_hat = x_0.reshape((n, 1))

        # Load P0 and ensure it is appropriately sized
        P_0 = kwargs.pop('P_0', None)
        if P_0 is None:
            self.P = np.eye(n)
        else:
            self.P = np.array(P_0)
            if (n, n) != self.P.shape:
                raise ValueError('P_0 must be an nxn numpy array.')
        #endregion

        #region : Load noise models and ensure their validity
        Q = kwargs.pop('Q', None)
        R = kwargs.pop('R', None)

        if (Q is None) or (R is None):
            raise ValueError(
                'Noise covariance matrices Q and R must both be specified.'
            )
        self.Q = np.array(Q)
        self.R = np.array(R)
        if (n, n) != self.Q.shape:
            raise ValueError('Q must be an nxn numpy array.')
        # XXX Size of R is not checked since number of outputs unknown
        #endregion

        #region : Load motion model, determine its type, and ensure validity
        A = kwargs.pop('A', None)
        B = kwargs.pop('B', None)
        f = kwargs.pop('f', None)
        J_f = kwargs.pop('J_f', None)

        if (A is not None) and (B is not None):
            if (f is not None) or (J_f is not None):
                raise ValueError(
                    'f and J_f should not be defined '
                    'if using a linear motion model.'
                )
            else:
                # Motion model is linear
                self.A = A
                self.B = B
                self.predict = self.__predict_linear

        elif (f is not None) and (J_f is not None):
            if not (callable(f) and callable(J_f)):
                raise ValueError("f and J_f must be functions.")
            elif (A is not None) or (B is not None):
                raise ValueError(
                    'A and B should not be defined '
                    'if using a nonlinear motion model.'
                )
            else:
                # Motion model is nonlinear
                self.f = f
                self.J_f = J_f
                self.predict = self.__predict_nonlinear
        else:
            raise ValueError(
                'Motion model not properly defined.'
                "For a linear motion model, provide matrices 'A' and 'B' "
                "For a nonlinear model, provide functions 'f' and 'J_f' "
                'where J_f is the jacobian of f.'
            )
        #endregion

        #region Load measurement model, determine type and ensure validity
        C = kwargs.pop('C', None)
        h = kwargs.pop('h', None)
        J_h = kwargs.pop('J_h', None)

        if not ((C is None) or (h is None) or (J_h is None)):
            raise ValueError(
                'C, h, and J_h cannot all be defined. '
                'Measurement model must be either linear or nonlinear.'
            )

        if C is not None:
            # Measurement model is linear
            self.C = C
            self.correct = self.__correct_linear
        elif (h is not None) and (J_h is not None):
            if not (callable(h) and callable(J_h)):
                raise ValueError("h and J_h must both be functions.")
            else:
                # Measurement model is nonlinear
                self.h = h
                self.J_h = J_h
                self.correct = self.__correct_nonlinear
        else:
            raise ValueError(
                'Measurement model not properly defined. '
                "For a linear measurement model, provide output matrix 'C'. "
                "For a nonlinear model, provide functions 'h' and 'J_h' "
                'where J_h is the jacobian of h.'
            )
        #endregion
        self.I = np.eye(n)

    #region Prediction Functions
    def predict(self, u, dt):
        """
        Predicts the next state of the system.

        Args:
            u: The control input.
            dt: The time step.
        """

        # Assigned in __init__ based on passed arguments
        pass

    def __predict_linear(self, u, dt):
        # Discretize A:
        A_d = self.I + self.A*dt

        self.x_hat = A_d@self.x_hat + dt*self.B*u
        self.P = A_d@self.P@np.transpose(A_d) + self.Q

    def __predict_nonlinear(self, u, dt):
        # Update Covariance Estimate
        F = self.J_f(self.x_hat, u, dt)
        self.P = F@self.P@np.transpose(F) + self.Q

        # Update State Estimate
        self.x_hat = self.f(self.x_hat, u, dt)
    #endregion

    #region Correction Functions
    def correct(self, y, dt=None):
        """
        Corrects the predicted state based on the measurement.

        Args:
            y: The measurement.
            dt: The time step.
        """

        # Assigned in __init__ based on passed arguments
        pass

    def __correct_linear(self, y, dt=None):
        y = np.squeeze(y)
        try:
            q = len(y)
        except TypeError:
            q = 1

        if q > 1:
            deletionList = np.nonzero(y == None)[0]
            y = np.delete(y, deletionList).reshape((q-len(deletionList), 1))
            C = np.delete(self.C, deletionList, axis=0)
            R = np.delete(self.R, deletionList, axis=0)
            R = np.delete(R, deletionList, axis=1)
        else:
            C = self.C
            R = self.R

        P_times_CTransposed = self.P @ np.transpose(C)
        S = C @ P_times_CTransposed + R
        K = P_times_CTransposed @ np.linalg.inv(S)

        try:
            m = len(y)
        except TypeError:
            m = 1
        if m > 1:
            self.x_hat = self.x_hat + K @  (y - C@self.x_hat)
        else:
            self.x_hat = self.x_hat + np.array(
                K * (y - C@self.x_hat),
                dtype=float
            )

        self.P = (self.I - K@C) @ self.P

    def __correct_nonlinear(self, y, dt=None):
        y = np.squeeze(y)
        try:
            q = len(y)
        except TypeError:
            q = 1

        z = self.h(self.x_hat, dt)
        H = self.J_h(self.x_hat, dt)

        if q > 1:
            deletionList = np.nonzero(y == None)[0]
            y = np.delete(y, deletionList).reshape((q-len(deletionList), 1))
            z = np.delete(z, deletionList).reshape((q-len(deletionList), 1))
            H = np.delete(H, deletionList, axis=0)
            R = np.delete(self.R, deletionList, axis=0)
            R = np.delete(R, deletionList, axis=1)
        else:
            R = self.R

        # Precompute Terms
        P_times_HTransposed = self.P @ np.transpose(H)

        S = H @ P_times_HTransposed + R
        K = P_times_HTransposed @ np.linalg.inv(S)

        # Update State Estimate
        self.x_hat = self.x_hat + K @ (y - z)

        # Update Covariance Estimate
        self.P = (self.I - K@H) @ self.P
    #endregion
    pass


class KalmanFilter(EKF):
    """Implements a linear Kalman filter.

    A Kalman filter is an optimal estimator that is used to estimate the
    state of a linear dynamic system. It is commonly used in signal processing,
    control systems, and navigation applications.

    This implementation assumes that both the motion and measurement models
    are linear. For nonlinear models, consider using an Extended Kalman Filter
    (EKF) instead.

    Args:
        A (ndarray): The state transition matrix.
        B (ndarray): The control-input matrix.
        C (ndarray): The output matrix.
        Q (ndarray): The process noise covariance matrix as a 2D numpy array.
        R (ndarray): The measurement noise covariance matrix as a 2D numpy
            array.
        x_0 (ndarray): The initial state estimate as a 1D or 2D numpy array.
        P_0 (ndarray): The initial covariance matrix as a 2D numpy array.

    Raises:
        AttributeError: If the motion and measurement models are not linear.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not (hasattr(self, "A")
                and hasattr(self, "B")
                and hasattr(self, "C")):
            raise AttributeError(
                'For a Kalman filter, both the motion and measurement '
                'models must be linear. For nonlinear models, try '
                'using an Extended Kalman Filter (EKF) instead.'
            )

class QCarEKF:
    """ An EKF designed to estimate the 2D position and orientation of a QCar.

    Attributes:
        kf (KalmanFilter): Kalman filter for orientation estimation.
        ekf (EKF): Extended Kalman filter for pose estimation.
        L (float): Wheelbase of the vehicle.
        x_hat (ndarray): State estimate vector [x; y; theta].
    """

    def __init__(
            self,
            x_0,
            Q_kf=np.diagflat([0.0001, 0.001]),
            R_kf=np.diagflat([.001]),
            Q_ekf=np.diagflat([0.01, 0.01, 0.01]),
            R_ekf=np.diagflat([0.01, 0.01, 0.001])
        ):
        """Initialize QCarEKF with initial state and noise covariance matrices.

        Args:
            x_0 (ndarray): Initial state vector [x, y, theta].
            Q_kf (ndarray, optional): KF process noise covariance matrix.
            R_kf (ndarray, optional): KF measurement noise covariance matrix.
            Q_ekf (ndarray, optional): EKF process noise covariance matrix.
            R_ekf (ndarray, optional): EKF measurement noise covariance matrix.
        """

        x_0 = np.squeeze(x_0)
        self.kf = KalmanFilter(
            x_0=[x_0[2], 0],
            P0=np.eye(2),
            Q=Q_kf,
            R=R_kf,
            A=np.array([[0, -1], [0, 0]]),
            B=np.array([[1], [0]]),
            C=np.array([[1, 0]])
        )

        self.ekf = EKF(
            x_0=x_0,
            P0=np.eye(3),
            Q=Q_ekf,
            R=R_ekf,
            f=self.f,
            J_f=self.J_f,
            C=np.eye(3)
        )

        self.L = 0.2
        self.x_hat = self.ekf.x_hat

    def f(self, x, u, dt):
        """Motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Updated state vector after applying motion model.
        """

        return x + dt * u[0] * np.array([
            [np.cos(x[2,0])],
            [np.sin(x[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    def J_f(self, x, u, dt):
        """Jacobian of the motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Jacobian matrix of the motion model.
        """

        return np.array([
            [1, 0, -dt*u[0]*np.sin(x[2,0])],
            [0, 1, dt*u[0]*np.cos(x[2,0])],
            [0, 0, 1]
        ])

    def update(self, u=None, dt=None, y_gps=None, y_imu=None):
        """Update the EKF state estimate using GPS and IMU measurements.

        Args:
            u (ndarray, optional): Control input vector [v, delta].
            dt (float, optional): Time step in seconds.
            y_gps (ndarray, optional): GPS measurement vector [x, y, th].
            y_imu (float, optional): IMU measurement of orientation.
        """

        if dt is not None:
            if y_imu is not None:
                self.kf.predict(y_imu, dt)
                self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])
            if u is not None:
                self.ekf.predict(u, dt)
                self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        if y_gps is not None:
            y_gps = np.squeeze(y_gps)

            y_kf = (
                wrap_to_pi(y_gps[2] - self.kf.x_hat[0,0])
                + self.kf.x_hat[0,0]
            )
            self.kf.correct(y_kf, dt)
            self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])

            y_ekf = np.array([
                [y_gps[0]],
                [y_gps[1]],
                [self.kf.x_hat[0,0]]
            ])
            z_ekf = y_ekf - self.ekf.C @ self.ekf.x_hat
            z_ekf[2] = wrap_to_pi(z_ekf[2])
            y_ekf = z_ekf + self.ekf.C @ self.ekf.x_hat
            self.ekf.correct(y_ekf, dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        else:
            y_ekf = (
                wrap_to_pi(self.kf.x_hat[0,0] - self.ekf.x_hat[2,0])
                + self.ekf.x_hat[2,0]
            )
            self.ekf.correct([None, None, y_ekf], dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        self.x_hat = self.ekf.x_hat