import React from "react"
import {Container, Row, Col} from 'react-bootstrap'

import "./style.css"
import fau_logo from './images/fau_logo.png';
import pr_lab_logo from './images/pr_lab_logo.png';


class Header extends React.Component{

  render(){
    return(
        <div className="title-container">
          <Row fluid="true">
            <Col xs={2} md={3}>
              <img src={fau_logo} alt="FAU Logo" className="image-logo"/>
            </Col>
            <Col xs={8} md={6}>
              <div className="main-title">
                {this.props.title}
              </div>
            </Col>
            <Col xs={2} md={3}>
              <img src={pr_lab_logo} alt="Pattern Recognition Lab Logo" className="image-logo"/>
            </Col>
          </Row>
        </div>
  )}
}

export default Header;

//
