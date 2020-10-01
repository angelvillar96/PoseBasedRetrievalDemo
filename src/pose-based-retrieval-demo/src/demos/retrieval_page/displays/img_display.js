import React from "react"

import Text from '../titles/text_display'
import "./styles/display_styles.css"

class ImgDisplay extends React.Component{

  constructor(props){
    super(props)
    this.state = {
        file: undefined,
        file_url: "",
        file_name: "No Image Loaded"
    }
    this.refresh = this.refresh.bind(this)
    this.update_display = this.update_display.bind(this)
  }


  // updating the display if the state has change
  refresh(){
    if(this.props.file_name === this.state.file_name &&
       this.props.file_url === this.state.file_url ){
      return
    }else{
      this.update_display()
    }
  }

  // updating the image content and title
  async update_display(){
    if(this.props.file_url.length === 0){
      return
    }

    await this.setState({
      file: URL.createObjectURL(this.props.file),
      file_url: this.props.file_url,
      file_name: this.props.file_name
    })
  }

  render(){
    this.refresh()
    return(
      <div className="img_display_area">
        <div>
          <Text text_display={this.state.file_name} font_size="1.6em" font_style="normal"/>
        </div>
        <div className="img_display" style={{backgroundImage: "url("+this.state.file+")"}}>
        </div>
      </div>
    )
  }

}

export default ImgDisplay

//
