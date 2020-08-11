import React from "react"
import {Container, Row, Col} from 'react-bootstrap'
import Button from 'react-bootstrap/Button'

import InputImg from "./input_img.js"
import ImgDisplay from "../displays/img_display.js"

import "./styles/input_styles.css"

class InputArea extends React.Component{
  constructor(props){
    super(props)
    this.state = {
        file: undefined,
        file_url: "",
        file_name: ""
    }
    this.update_state = this.update_state.bind(this)
  }

  // method that updates the state when a child component is changed
  update_state(state_id, value){
    this.setState({
      [state_id]:value
    });
  }


  render(){
    return(
      <Container className="input_area">
        <Row fluid="true">
          <Col md={2}>
          </Col>
          <Col sm={12} md={8}>
            <Row fluid="true">
              <ImgDisplay file={this.state.file} file_url={this.state.file_url}
                          file_name={this.state.file_name}/>
            </Row>
            <Row className="buttons_area">
              <Col sm={1} md={1}></Col>
              <Col sm={4} md={4}>
                <InputImg className="myButton" update_state={this.update_state}/>
              </Col>
              <Col sm={1} md={2}></Col>
              <Col sm={4} md={4}>
                <Button className="myButton" variant="primary">Process Image</Button>
              </Col>
              <Col sm={1} md={1}></Col>
            </Row>
          </Col>
          <Col md={2}>
          </Col>
        </Row>
      </Container>
    )
  }

}

export default InputArea

// <Row>
  // <DropFile/>
// </Row>
// <Row className="buttons_area">
  // <Button variant="primary">Process Image</Button>
// </Row>
