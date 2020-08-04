import React from "react"
import {Container, Row, Col} from 'react-bootstrap'
import Button from 'react-bootstrap/Button'

import DropFile from "./input_img.js"
import ImgDisplay from "../displays/img_display.js"

import "./styles/input_styles.css"

class InputArea extends React.Component{


    render(){
      return(
        <Container className="input_area">
          <Row fluid="true">
            <Col sm={12} md={4}>
              <Row>
                <DropFile/>
              </Row>
              <Row className="buttons_area">
                <Button variant="primary">Process Image</Button>
              </Row>
            </Col>
            <Col sm={12} md={8}>
              <ImgDisplay/>
            </Col>
          </Row>
        </Container>
      )
    }

}

export default InputArea
