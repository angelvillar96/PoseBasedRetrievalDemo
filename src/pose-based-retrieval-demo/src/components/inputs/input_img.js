import React from "react"
// import Text from "./text_display.js"

class DropFile extends React.Component{

  // class constructor
  constructor(props){
    super(props)
    this.state = {
      file: undefined,
      file_name: "",
      message: "Drop an image %aux% or %aux% Click to add"
    }

    this.on_change = this.on_change.bind(this)
  }

  // on change method
  on_change(e){

    let new_file = e.target.files[0]
    let fname = e.target.value
    fname = fname.split("\\")
    fname = fname[fname.length-1]

    if( new_file !== undefined && fname.length>0 ){
      this.setState({
        file: new_file,
        file_name: fname,
        message: "Loaded file: %aux% " + fname
      })
    }else{
      this.setState({
        file: undefined,
        file_name: "",
        message: "Drop an image %aux% or %aux% Click to add"
      })
    }

  }


  // render HTML
  render(){
    return(
      <div className="dropZoneContainer file_input">
        <input type="file" id="drop_zone" className="FileUpload" accept=".jpg,.png,.jpeg,.gif" onChange={this.on_change}/>
        <div className="dropZoneOverlay">
          <div className="dropZoneText">
            TODO
            {/*<Text text_display={this.state.message}/>*/}
          </div>
        </div>
      </div>
    )
  }

}

export default DropFile
