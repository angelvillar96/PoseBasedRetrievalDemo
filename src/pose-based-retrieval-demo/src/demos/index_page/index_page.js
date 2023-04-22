/*
Index page where the user selects the demo that he wants to run

@author: Angel Villar-Corrales
*/

import React from 'react';
import {Container, Row, Col} from 'react-bootstrap'

import Header from '../../components/pr_lab_header/title.js'
import Separator from '../../components/separator/separator.js'
import CustomCard from "../../components/card/card.js"
import DetectionLogo from './imgs/DetectionDemo.png';
import PoseEstimationLogo from './imgs/PoseEstimationDemo.png';
import ObjectDetectionLogo from './imgs/ObjectDetectionDemo.png';
import IccLogo from './imgs/IccDemo.png'

class IndexPage extends React.Component{

  render(){

    const card1 = <CustomCard title="Pose-Based Retrieval Demo" route="/pose-based-retrieval"
                   image={PoseEstimationLogo}
                   description='Retrieving images from different datasets based on the
                                similarity between the poses of the person instances.'/>

    // const card2 = <CustomCard title="Person Detection Demo" route="/person-detection"
    //                image={DetectionLogo}
    //                description='Detecting relevant persons in
    //                             images from art-historical and archaeological datasets.'/>

    // const card3 = <CustomCard title="Object Detection Demo" route="/object-detection"
    //                image={ObjectDetectionLogo}
    //                description='Detecting relevant objects in
    //                             images from art-historical and archaeological datasets.'/>

    // const card4 = <CustomCard title="Image Composition Demo" route="/icc-based-retrieval"
    //                image={IccLogo}
    //                description='Retrieving images from art history dataset based on the
    //                compositions in images.'/>

    return (
      <div>
        <Header title={"Digital Humanities Demos"}/>
        <Separator/>
        <Container className="input_area" fluid>
          <Row fluid="true">
            <Col sm={1}></Col>
            <Col sm={2}>{card1}</Col>
            {/* <Col sm={2}>{card2}</Col>
            <Col sm={2}>{card3}</Col>
            <Col sm={2}>{card4}</Col> */}
          </Row>
        </Container>
      </div>
    );
  }
}

export default IndexPage;
