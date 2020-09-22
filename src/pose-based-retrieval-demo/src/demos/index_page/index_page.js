/*
Index page where the user selects the demo that he wants to run

@author: Angel Villar-Corrales
*/

import React from 'react';
import {Container, Row, Col} from 'react-bootstrap'

import CustomCard from "../../components/card/card.js"
import DetectionLogo from './imgs/DetectionDemo.png';
import PoseEstimationLogo from './imgs/PoseEstimationDemo.png';

class IndexPage extends React.Component{

  render(){

    const card1 = <CustomCard title="Pose-Based Retrieval Demo" route="/pose-based-retrieval"
                   image={PoseEstimationLogo}
                   description='Retrieving images from different datasets based on the
                                similiraty between the poses of the person instances.'/>

    const card2 = <CustomCard title="Object Detection Demo" route="/"
                   image={DetectionLogo}
                   description='Detecting different relevant objects and actions in
                                images from artistical and archeological datasets.'/>

    return (
      <Container className="input_area">
        <Row fluid="true">
          <Col>{card1}</Col>
          <Col>{card2}</Col>
        </Row>
        <Row fluid="true">
          <Col><div/></Col>
          <Col><div/></Col>
        </Row>
      </Container>
    );
  }
}

export default IndexPage;
