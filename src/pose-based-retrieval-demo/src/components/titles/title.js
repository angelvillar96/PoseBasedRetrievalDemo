import React from "react"
import {Container, Row, Col} from 'react-bootstrap'

import "./styles/titles.css"
import fau_logo from './images/fau_logo.png';
import pr_lab_logo from './images/pr_lab_logo.png';


class Title extends React.Component{

  render(){
    return(
        <Container className="title-container">
          <Row fluid="true">
            <Col xs={2} md={3}>
              <img src={fau_logo} alt="FAU Logo" className="image-logo"/>
            </Col>
            <Col xs={8} md={6}>
              <div className="main-title">
                Pose-Based Image Retrieval Demo
              </div>
            </Col>
            <Col xs={2} md={3}>
              <img src={pr_lab_logo} alt="Pattern Recognition Lab Logo" className="image-logo"/>
            </Col>
          </Row>
        </Container>
  )}
}

export default Title;

//
