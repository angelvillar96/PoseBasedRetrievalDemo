import React from "react"

import Button from 'react-bootstrap/Button'

import "./styles/input_styles.css"
import camera from '../../lib/camera.js'
import {objectUrl2File, get_timestamp} from '../../lib/utils.js'


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
    this.props.update_state("display_name", this.state.file_name)

    this.on_click = this.on_click.bind(this)
    this.set_img_uri = this.set_img_uri.bind(this)
  }

  // processing the click on the take image button
  on_click(){
    console.log(camera)
    camera.start_camera(this.set_img_uri)
  }


  // method for updating the uri from the img taken with the camera
  async set_img_uri(img_url){

    let img_file = objectUrl2File(img_url)
    await this.setState({
        "file": img_file,
        "file_url": img_url,
        "file_name": "camera_img_" + get_timestamp() + ".png"
    });

    this.props.update_state("file", this.state.file)
    this.props.update_state("file_url", this.state.file_url)
    this.props.update_state("file_name", this.state.file_name)
    this.props.update_state("display_name", this.state.file_name)
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
