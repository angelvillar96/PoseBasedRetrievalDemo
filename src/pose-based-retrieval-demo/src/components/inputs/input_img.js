import React from "react"

import Button from 'react-bootstrap/Button'

import "./styles/input_styles.css"

class InputImg extends React.Component{

  // class constructor
  constructor(props){
    super(props)
    this.state = {
      file: undefined,
      file_url: "",
      file_name: "",
    }

    this.props.update_state("file", this.state.file)
    this.props.update_state("file_url", this.state.file_url)
    this.props.update_state("file_name", this.state.file_name)
    this.inputReference = React.createRef();
    this.on_change = this.on_change.bind(this)
    this.on_click = this.on_click.bind(this)
  }

  // processing the click on the Load Img button
  async on_click(){
    this.inputReference.current.click();
  }

  // processing the load of a new image
  on_change(e){

    let new_file = e.target.files[0]
    let file_url = e.target.value
    let fname = file_url.split("\\")
    fname = fname[fname.length-1]
    if( new_file !== undefined && fname.length>0 ){
      this.setState({
        file: new_file,
        file_url: file_url,
        file_name: fname
      })
    }else{
      this.setState({
        file: undefined,
        file_url: "",
        file_name: ""
      })
    }
    e.target.value = null
    this.props.update_state("file", new_file)
    this.props.update_state("file_url", file_url)
    this.props.update_state("file_name", fname)
    this.props.update_state("display_name", fname)
  }

  // render HTML
  render(){
    return(
      <div>
        <input style={{display:"none"}} ref={this.inputReference} type="file" accept=".jpg,.jpeg,.png,.tif" onChange={this.on_change}/>
        <Button variant="primary" onClick={this.on_click}>Load Image</Button>
      </div>
    )
  }

}

export default InputImg
