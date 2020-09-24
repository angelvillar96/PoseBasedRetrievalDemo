/*
Index page where the user selects the demo that he wants to run

@author: Angel Villar-Corrales
*/

import React from 'react';
import {Container, Row, Col} from 'react-bootstrap'

import Header from '../../components/pr_lab_header/title.js'
import Separator from '../../components/separator/separator.js'
import Text from '../../components/text_display/text_display.js'
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
      <div>
        <Header title={"Digital Humanities Demos"}/>
        <Separator/>
        <Container className="input_area">
          <Row fluid="true">
            <Col sm={2}></Col>
            <Col sm={4}>{card1}</Col>
            <Col sm={1}></Col>
            <Col sm={4}>{card2}</Col>
            <Col sm={1}></Col>
          </Row>
          <Row fluid="true">
            <Col><div/></Col>
            <Col><div/></Col>
          </Row>
        </Container>
        <Separator/>
        <Text text_display="by Angel Villar-Corrales" font_size="1.3em" align="center"/>
      </div>
    );
  }
}

export default IndexPage;
