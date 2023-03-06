//
// Created by yt on 17.11.20.
//
#pragma once

#include <stdlib.h>
#include <random>
#include <cstdint>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <cmath>
#include "Eigen/Dense"


namespace raisim {

    /** This environment class implements n equally setup parallel environmets. Random values are generated independently
     *  for each environment.
     **/
    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0., 1.), gen_(rd_()) {

	        /// create world
    	    world_ = std::make_unique<raisim::World>();

            /// add objects
            /// robot properties are defined in the urdf file
            robot_ = world_->addArticulatedSystem(
                    resourceDir_ + "/pointmass/urdf/pointmass.urdf");
            robot_->setName("pointmass");
            robot_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);
            world_->addGround(-0.101);

            /// get robot data
            gcDim_ = robot_->getGeneralizedCoordinateDim();
            gvDim_ = robot_->getDOF();
            nRotors_ = gvDim_ - 6; //4
            /** observation space: 18 entries
             *                      position: 3 entr.
             *                      orientation: 9 entr. (orientation matrix R)
             *                      velocity: 3 entr.
             *                      angular velocity: 3 entr.
             *  add. entries:       control signal of the pid controller: 4 entr.
             *
             **/
            obDim_ = 18;
            actionDim_ = 1;
            probDim_ = 1;

            /// initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); genForces_.setZero(gvDim_); /// convention described on top
            actionMean_=0; actionStd_=0;
            obDouble_.setZero(obDim_); targetPoint_.setZero(obDim_);

            /// nominal configuration of the robot: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
            gc_init_ << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            /// constant rotor velocity for visualization purposes
            gv_init_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

            /// initialize rotor thrusts_ and conversion matrix for generated forces and torques
            thrusts_=0; controlThrusts_=0;
            /// conversion matrix - converts the control input to generated force and torques from the motor
            thrusts2TorquesAndForces_ = 1.0;

            /** action scaling:
             *  action is scaled down to [-1,1] for the training of the neural network. Thus, the neural network
             *  determines the action within a range of [-1,1]
             **/
            actionMean_=0; actionStd_=1;

            /// indices of links that should not make contact with ground - all links rotors
            bodyIndices_.insert(robot_->getBodyIdx("rotor_0"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_1"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_2"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_3"));

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile(cfg["reward"]);

            /// set gravity to 0
            gravity_[2]=0;  world_->setGravity(gravity_); 

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(robot_);
                visPoint = server_->addVisualSphere("visPoint", 0.2, 0.8, 0, 0);
            }

            Yaml::ConstIterator key;
            for(key = cfg.Begin(); key != cfg.End(); key++) {
                if ((*key).first == "random_robot_state") {
                    random_robot_state_= (*key).second.As<bool>();
                } else if ((*key).first == "random_target_state"){
                    random_target_state_= (*key).second.As<bool>();
                }
            }

        }


        void init() final {}

        void reset() final {
        /// set random target point or state
            if (random_robot_state_==true) {
                setRandomStates(0, true, 2);
            }
            if (random_target_state_ == true)
                setRandomTargets(10, update_every_n_episode_);
            else
                setTarget(5, 0, 0);


            robot_->setState(gc_init_, gv_init_);
            updateObservation();

            if (visualizable_) server_->focusOn(robot_);
        }

        float step(const Eigen::Ref<EigenVec> &action) final {

            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
                if (server_) server_->lockVisualizationServerMutex();
                thrusts_=controlThrusts_;
                applyThrusts();
                world_->integrate();
                if (server_) server_->unlockVisualizationServerMutex();
            }

            /// get the control signal from the neural network
            controlThrusts_ = action[0];

            /// scale up bounded action input to actual rotor control signals
            controlThrusts_ = controlThrusts_*actionStd_;
            controlThrusts_ += actionMean_;

            updateObservation();

            /// relAbsPosition is used for the reward function
            relativeAbsPosition = (targetPoint_.head(3) - bodyPos_).norm();

            /// set a zone (sphere near the target) with success rewards.
            success = (relativeAbsPosition<1) ? 1 : 0;

            /// reward function
            rewards_.record("success", success);

            return rewards_.sum();
        }

        void updateObservation() {
            /// get robot state and transform specific components to the body frame
            robot_->getBaseOrientation(worldRot_);
            robot_->getState(gc_, gv_);
            bodyPos_ = gc_.head(3);
            bodyRot_ = worldRot_.e().transpose();
            bodyLinVel_ = bodyRot_ * gv_.segment(0, 3);
            bodyAngVel_ = bodyRot_ * gv_.segment(3, 3);
            robot_->getBaseOrientation(quat_);

            /// observation vector: observation space and control signal which are passed on to the RL-algorithm
            for (size_t i = 0; i < 3; i++) {
                obDouble_[i] = bodyPos_[i];
            }

            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    obDouble_[j + (i + 1) * 3] = bodyRot_(i, j);
                }
                obDouble_[i + 12] = bodyLinVel_[i];
                obDouble_[i + 15] = bodyAngVel_[i];
            }
            obDouble_ -= targetPoint_;

            /// add Gaussian noise
            for (size_t i = 0; i < obDim_; i++) {
                obDouble_[i] *= (1 + 0.05 * normDist_(gen_));
            }

        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float &terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            for (auto &contact: robot_->getContacts()) {
                if (bodyIndices_.find(contact.getlocalBodyIndex()) == bodyIndices_.end()) {
                    return true;
                }
            }

            for (int i = 0; i < 3; i++) {
                if (abs(bodyPos_[i]) > 20) {
                    return true;
                }
            }
            terminalReward = 0.f;
            return false;
        }

        void applyThrusts() {
            /// calculate forces and torques w.r.t. the center of mass
            torquesAndForces_ = thrusts2TorquesAndForces_ * thrusts_;
            forces_baseFrame_ << torquesAndForces_, 0.0, 0.0;
            torques_baseFrame_ << 0.0, 0.0, 0.0;

            torques_worldFrame_.e() = worldRot_.e() * torques_baseFrame_;
            forces_worldFrame_.e() = worldRot_.e() * forces_baseFrame_;

            /// apply forces and torques to the robot
            genForces_.head(3) = forces_worldFrame_.e();
            genForces_.segment(3, 3) = torques_worldFrame_.e();
            robot_->setGeneralizedForce(genForces_);

            /// this option will visualize the applied forces and torques
            robot_->setExternalForce(0, forces_worldFrame_);
            // robot_->setExternalTorque(0, torques_worldFrame_);
        }


        /********* Custom methods **********/
        void setTarget(double x, double y, double z) {
            targetPoint_[0] = x;
            targetPoint_[1] = y;
            targetPoint_[2] = z;
            if (visualizable_) visPoint->setPosition(targetPoint_.head(3));
        }

        void setRandomTargets(double radius, int update_every_n_episode) {
            if (loopCount_ <= 0) {
                for (int i = 0; i < probDim_; i++) targetPoint_(i) = normDist_(gen_);
                targetPoint_.head(3) /= targetPoint_.head(3).norm();
                targetPoint_.head(3) *= radius; // target point has distance of 10 m within a sphere
                if (visualizable_) visPoint->setPosition(targetPoint_.head(3));
                loopCount_ = update_every_n_episode;
            }
            loopCount_--;
        }

        void setRandomStates(double pos, bool rot_bool, double vel) {
            for (int i = 0; i < probDim_; i++) {
                gc_init_(i) = pos * normDist_(gen_);
                gv_init_(i) = vel * normDist_(gen_);
            }
        }

        void calculateEulerAngles() {
            double sinr_cosp = 2 * (quat_[0] * quat_[1] + quat_[2] * quat_[3]);
            double cosr_cosp = 1 - 2 * (quat_[1] * quat_[1] + quat_[2] * quat_[2]);
            eulerAngles_[0] = std::atan2(sinr_cosp, cosr_cosp);

            // pitch (y-axis rotation)
            double sinp = 2 * (quat_[0] * quat_[2] - quat_[3] * quat_[1]);
            if (std::abs(sinp) >= 1)
                eulerAngles_[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
            else
                eulerAngles_[1] = std::asin(sinp);

            // yaw (z-axis rotation)
            double siny_cosp = 2 * (quat_[0] * quat_[3] + quat_[1] * quat_[2]);
            double cosy_cosp = 1 - 2 * (quat_[2] * quat_[2] + quat_[3] * quat_[3]);
            eulerAngles_[2] = std::atan2(siny_cosp, cosy_cosp);

            for (int i; i < 3; i++) {
                eulerAngles_[i] = std::abs(eulerAngles_[i]) + 1e-4;
            }
        }

        /// environment related variables
        bool visualizable_ = true;
        raisim::ArticulatedSystem *robot_;
        raisim::Visuals *visPoint;
        raisim::Vec<3> gravity_;

        int gcDim_, gvDim_, nRotors_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_;
        Eigen::VectorXd obDouble_, targetPoint_;
        double actionMean_, actionStd_;

        raisim::Mat<3, 3> worldRot_;
        raisim::Vec<4> quat_;
        Eigen::Vector3d bodyPos_, bodyLinVel_, bodyAngVel_;
        Eigen::Matrix3d bodyRot_;

        double thrusts_, controlThrusts_;
        double thrusts2TorquesAndForces_;
        double torquesAndForces_;
        Eigen::Vector3d torques_baseFrame_, forces_baseFrame_;
        raisim::Vec<3> torques_worldFrame_, forces_worldFrame_;
        Eigen::VectorXd genForces_;
        Eigen::Vector3d eulerAngles_;

        /// model parameters
        const double motConst = 20, thrConst = 8.54858e-06;

        /// reward related variables
        raisim::Reward rewards_;
        std::set<size_t> bodyIndices_;
        double terminalRewardCoeff_ = -15.;
        double relativeAbsPosition;
        double success;

        /// other variables
        int update_every_n_episode_=2;
        int loopCount_ = update_every_n_episode_;
        int probDim_;
        bool random_robot_state_ = false, random_target_state_=false;

        std::normal_distribution<double> normDist_;
        std::random_device rd_;
        std::mt19937 gen_;

    };
}

