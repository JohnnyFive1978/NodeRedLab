// Nodes/Example/example.c
/**
 * @file example.c
 * @brief Kalman filter implementation for state estimation.
 *
 * This file demonstrates a basic Kalman filter used for estimating the state of a system
 * from noisy measurements. It initializes the filter with given parameters, predicts the
 * state, and updates it based on new measurements.
 *
// Nodes/Example/example.c
/**
 * @file example.c
 * @brief Kalman filter implementation for state estimation.
 * 
 * This file demonstrates a basic Kalman filter used for estimating the state of a system
 * from noisy measurements. It initializes the filter with given parameters, predicts the
 * state, and updates it based on new measurements.
 *
 * @author [Your Name]
 * @date [Current Date]
 * @version 1.0
 */

 #include <stdio.h>

 /**
  * @def N
  * @brief Number of dimensions in the state vector (e.g., position and velocity).
  */
 #define N 2
 
 /**
  * @struct KalmanParams
  * @brief Structure to hold Kalman filter parameters.
  *
  * @var A State transition matrix.
  * @var H Observation matrix.
  * @var Q Process noise covariance matrix.
  * @var R Measurement noise variance.
  */
 typedef struct {
     float A[N][N]; /**< State transition matrix. */
     float H[1][N]; /**< Observation matrix. */
     float Q[N][N]; /**< Process noise covariance matrix. */
     float R;       /**< Measurement noise variance. */
 } KalmanParams;
 
 /**
  * @struct KalmanState
  * @brief Structure to hold the current Kalman filter state.
  *
  * @var x State vector (e.g., position, velocity).
  * @var P Covariance matrix of the state estimation error.
  */
 typedef struct {
     float x[N];  /**< State vector. */
     float P[N][N]; /**< Covariance matrix. */
 } KalmanState;
 
 /**
  * Global instances of Kalman filter parameters and state.
  */
 KalmanParams kalman_params;
 KalmanState kalman_state;
 
 /**
  * @brief Initializes the Kalman filter with given parameters.
  *
  * Sets up the Kalman filter's matrices (A, H, Q) and measurement noise variance (R),
  * and initializes the state vector and covariance matrix to identity for simplicity.
  *
  * @param A_ State transition matrix.
  * @param H_ Observation matrix.
  * @param Q_ Process noise covariance matrix.
  * @param R_ Measurement noise variance.
  */
 void init_kalman_filter(float A_[N][N], float H_[1][N], float Q_[N][N], float R_) {
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             kalman_params.A[i][j] = A_[i][j];
             kalman_params.Q[i][j] = Q_[i][j];
             if (i == j) {
                 kalman_state.P[i][j] = 1.0f;
             } else {
                 kalman_state.P[i][j] = 0.0f;
             }
         }
     }
 
     for (int i = 0; i < N; i++) {
         kalman_params.H[0][i] = H_[0][i];
     }
 
     kalman_params.R = R_;
 }
 
 /**
  * @brief Predicts the state and covariance matrix based on the current state.
  *
  * Updates the state vector (x) by applying the state transition matrix (A) to
  * the current state, then updates the covariance matrix (P) by predicting its
  * future value considering the process noise covariance (Q).
  */
 void predict() {
     // Predict the state
     float x_pred[N];
     for (int i = 0; i < N; i++) {
         x_pred[i] = 0.0f;
         for (int j = 0; j < N; j++) {
             x_pred[i] += kalman_params.A[i][j] * kalman_state.x[j];
         }
     }
 
     // Predict the covariance
     float P_pred[N][N];
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             P_pred[i][j] = 0.0f;
             for (int k = 0; k < N; k++) {
                 P_pred[i][j] += kalman_params.A[i][k] * kalman_state.P[k][j];
             }
         }
     }
 
     // Add process noise
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             P_pred[i][j] += kalman_params.Q[i][j];
         }
     }
 
     // Update state and covariance
     for (int i = 0; i < N; i++) {
         kalman_state.x[i] = x_pred[i];
     }
 
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             kalman_state.P[i][j] = P_pred[i][j];
         }
     }
 }
 
 /**
  * @brief Updates the state and covariance matrix based on a new measurement.
  *
  * Calculates the Kalman gain, then updates the state vector (x) using this gain
  * to incorporate the information from the measurement (z), and finally updates
  * the covariance matrix (P) considering the Kalman gain and measurement noise.
  *
  * @param z New measurement.
  */
 void update(float z) {
     // Calculate Kalman gain
     float S = 0.0f;
     for (int i = 0; i < N; i++) {
         S += kalman_params.H[0][i] * kalman_state.P[i][0] * kalman_params.H[0][i];
     }
     S += kalman_params.R;
 
     float K[N];
     for (int i = 0; i < N; i++) {
         K[i] = kalman_state.P[i][0] * kalman_params.H[0][i] / S;
     }
 
     // Update state
     float y = z - kalman_params.H[0][0] * kalman_state.x[0]; // Measurement residual
     for (int i = 0; i < N; i++) {
         kalman_state.x[i] += K[i] * y;
     }
 
     // Update covariance
     float I_minus_KH[N][N];
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             if (i == j) {
                 I_minus_KH[i][j] = 1.0f - K[i] * kalman_params.H[0][j];
             } else {
                 I_minus_KH[i][j] = -K[i] * kalman_params.H[0][j];
             }
         }
     }
 
     float P_updated[N][N];
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             P_updated[i][j] = 0.0f;
             for (int k = 0; k < N; k++) {
                 P_updated[i][j] += I_minus_KH[i][k] * kalman_state.P[k][j];
             }
         }
     }
 
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             kalman_state.P[i][j] = P_updated[i][j];
         }
     }
 }
 
 /**
  * @brief Prints the current state vector and covariance matrix.
  */
 void print_state() {
     printf("State:\n");
     for (int i = 0; i < N; i++) {
         printf("x[%d] = %f\n", i, kalman_state.x[i]);
     }
 
     printf("\nCovariance Matrix:\n");
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             printf("P[%d][%d] = %f ", i, j, kalman_state.P[i][j]);
         }
         printf("\n");
     }
 }
 
 /**
  * @brief Main function to demonstrate the Kalman filter's usage.
  *
  * Initializes the Kalman filter, simulates some measurements, predicts and updates
  * the state, then prints the final state.
  */
 int main() {
     // Initialize the Kalman filter
     float A_[N][N] = {{1.0f, 1.0f}, {0.0f, 1.0f}};
     float H_[1][N] = {{1.0f, 0.0f}};
     float Q_[N][N] = {{0.5f, 0.0f}, {0.0f, 0.5f}};
     float R_ = 1.0f;
 
     init_kalman_filter(A_, H_, Q_, R_);
 
     // Initial state
     kalman_state.x[0] = 0.0f; // initial position estimate
     kalman_state.x[1] = 0.5f; // initial velocity estimate
 
     printf("Initial State:\n");
     print_state();
 
     // Simulate the Kalman filter with some measurements
     float z[] = {2.4, 3.2, 5.6, 7.8}; // example measurements
 
     for (int i = 0; i < sizeof(z) / sizeof(z[0]); i++) {
         predict();
         update(z[i]);
         printf("\nAfter measurement %d:\n", i + 1);
         print_state();
     }
 
     return 0;
 }