import React from "react"

import Button from 'react-bootstrap/Button'

import "./styles/input_styles.css"
import camera from '../../lib/camera.js'


class TakePicture extends React.Component{

  // class constructor
  constructor(props){
    super(props)
    this.state = {
      file: undefined,
      file_url: "",
      file_name: ""
    }

    this.props.update_state("file", this.state.file)
    this.props.update_state("file_url", this.state.file_url)
    this.props.update_state("file_name", this.state.file_name)

    this.on_click = this.on_click.bind(this)
  }

  // processing the click on the take image button
  on_click(){
    console.log(camera)
    camera.start_camera()
    img_uri = camera.get_img_uri()
    console.log("Off")
  }

  render(){
    return(
      <div>
        <Button variant="primary" onClick={this.on_click}>Take Picture</Button>
      </div>
    )
  }

}

export default TakePicture

//
